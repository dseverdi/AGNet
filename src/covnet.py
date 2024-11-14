
import os
import pickle
import numpy as np
import seaborn as sns

import pandas as pd

# randomness
import random

## nn
import torch 
import torch.nn as nn
import torch.optim as optim

# dataloader
from torch.utils.data   import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# argument parser
import argparse

# use AGMnet scripts
from data_types import VisSample 
from demo       import polygon_demo2

  
# debugging
import pdb
import tqdm


# visualization
import matplotlib.pyplot as plt

# evaluation metrics
from torcheval.metrics.functional import r2_score


###### flags ########

# define device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") 

verbose = True


class Params(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

####### model ##################

class Encoder(nn.Module):
    def __init__(self,input_size = 3, hidden_size=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size, num_layers=3, dropout=0.5, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self,x,x_lens):
        x_packed = pack_padded_sequence(x,x_lens, batch_first=True,enforce_sorted=False)
        output_packed, (h_n,c_n) = self.lstm(x_packed)
        return output_packed, (h_n, c_n)


class Aggregator(nn.Module):
    def __init__(self,hidden_size=10):
        super().__init__()
        self.aggregate =  nn.Sequential(
            nn.Linear(hidden_size,1),
            nn.Sigmoid()
        )
        self.hidden_size = hidden_size

    def forward(self,x):  
        output = self.aggregate(x)
        return output
    

class CovNet(nn.Module):
    def __init__(self, model_args : Params):
        super().__init__()
        self.input_size = model_args.input_size
        self.hidden_size = model_args.hidden_size
        
        # components
        self.poly_encoder = Encoder(self.input_size,self.hidden_size)
        self.sol_encoder = Encoder(self.input_size,self.hidden_size)
        self.aggregator  = Aggregator(2*self.hidden_size)

    def forward(self, x,x_lens, y, y_lens):
       # polygon encoder
        _,(h_x,_) = self.poly_encoder(x,x_lens)
                            
        # solution encoder     
        _,(h_y,_) = self.sol_encoder(y, y_lens) 
               
        # aggregation
        h_xy = torch.cat((h_x[-1],h_y[-1]),dim=1)

        
        output = self.aggregator(h_xy) 

        return output
    

#############################

def generate_instances(name,sol_sample=10):
    """
    generate instances
    """
    
    print(f'Reading instance from {name} taking up to {sol_sample} solutions ...')
    instances = VisSample.read_samples(name,sol_sample=sol_sample,normalize=True)
    
    # generate new instances
    new_instances = []
    
    # not solved
    not_solved = []
        
    # generate copy of instances where set of guards of arbitrary size are sampled from optimal guards
    for instance in instances:                
        # create not optimal solutions
        opt = len(instance.guards)
        if opt == 0:
            not_solved.append(instance.name)
            continue
            
        indx = sorted(random.sample(range(opt),random.randint(1,opt)))
        sampled_guards = instance.guards[indx]    

        new_instances.append(VisSample(name=instance.name,points = instance.points, guards = sampled_guards))

    print(f'Total of {len(not_solved)} not solved instances of {len(instances)}...')
    # evaluate coverage on instances
    coverage = [0] * len(new_instances)

    for i,instance in enumerate(new_instances):       

        try:
            _,_,_, _ ,coverage[i] = polygon_demo2(instance.guards,instance)
        except Exception as e:
            print(f'{instance.name}: ',e)
            continue
             
        
    print(f'Coverage for partial solution computed ...')
    
    return new_instances, coverage



def train_valid_test_split(data):
    # split to train and test

    print('Splitting data to train-valid-test ...')
    from sklearn.model_selection import train_test_split
    new_instances , coverage = data[0], data[1]

    # train, test split
    train_data, test_data, train_score, test_score = train_test_split(new_instances,coverage,test_size = 0.2,random_state=42)
    # train, valid split
    train_data, valid_data, train_score, valid_score = train_test_split(train_data,train_score, test_size = 0.1, random_state=42)    

    # training set
    train_set = [
        [
            torch.tensor([x.tolist() for x in train_data[i].points]),
            train_data[i].guards, 
            torch.tensor(train_score[i])
        ] for i in range(len(train_data))
        ]

    # vaid set
    valid_set = [
        [
            torch.tensor([x.tolist() for x in valid_data[i].points]),
            valid_data[i].guards,
            torch.tensor(valid_score[i])
        ] for i in range(len(valid_data))]

    # test set
    test_set = [
        [
            torch.tensor([x.tolist() for x in test_data[i].points]),
            test_data[i].guards,
            torch.tensor(test_score[i])
        ] for i in range(len(test_data))]
    

    print(f'* Train set: {len(train_set)}, valid set: {len(valid_set)}  Test set: {len(test_set)}')
    return train_set, valid_set, test_set


def collate_fn_padded(batch):
    """
        collate function to structure
    """
    
    sequences,positions,scores = zip(*batch)    
    
    # compute lengths of input   
    sequences_lens = [seq.shape[0] for seq in sequences]

    # batch_size
    batch_size = len(batch)

    # converting to tensor
    sequences = nn.utils.rnn.pad_sequence(sequences, batch_first = True, padding_value=-1)
    sols      = nn.utils.rnn.pad_sequence(positions, batch_first = True, padding_value=-1)    

    
    # compute points
    sols = torch.tensor(     
        [sequences[b][positions[b][positions[b] > -1]].numpy() for b in range(sequences.shape[0])]
    )

    # scores
    scores = torch.tensor([x for x in scores]).view(batch_size,1)

       
    # ToDo: make packed sequences
    return sequences, sols, scores


def collate_fn_packed(batch):
    """
        collate function to structure
    """
    
    sequences,positions,scores = zip(*batch)    
    
    # compute lengths of input       
    batch_size = len(batch)
    seq_lens   = [seq.shape[0] for seq in sequences]
    sol_lens   = [pos.shape[0] for pos in positions]

    # converting to tensor
    # pad sequences
    seqs = nn.utils.rnn.pad_sequence(sequences, batch_first = True, padding_value=-1)
    sols = nn.utils.rnn.pad_sequence([seqs[b][positions[b]] for b in range(batch_size)], batch_first = True, padding_value=-1)    
    
    # scores
    scores = torch.tensor([x for x in scores]).view(batch_size,1)

    #pdb.set_trace()
    # ToDo: make packed sequences
   

    return seqs, seq_lens, sols, sol_lens, scores
    

# training procedure
def train(model, train_generator, valid_generator, train_args):  

    print('PHASE: Training and validation')
    print('---------------------------------')    
    print(f'Total train samples: {len(train_generator)*train_generator.batch_size} Validation samples: {len(valid_generator)*train_generator.batch_size}')      
        # define optimizer
    optimizer = optim.AdamW(model.parameters(), lr = train_args.lr, weight_decay = train_args.wd)
    loss_fn = nn.MSELoss()

  
    # best validation loss
    best_val_loss = 1e9

    # number of epochs
    epochs = train_args.num_epochs

  
    # *** training
    for epoch in range(1,epochs+1):        
        # set model to train phase        
        model.train()

        train_loss = 0        
        for b, (x,x_len, y, y_len, score) in enumerate(train_generator):            
            optimizer.zero_grad()            
            # output from model 
            x, y, score = x.to(device), y.to(device), score.to(device)
            output = model(x,x_len,y,y_len) 
            # loss computation
            loss = loss_fn(output,score)
            # BPPT
            loss.backward()
            optimizer.step()
            # cummulate loss
            train_loss += loss.detach()


          # set model to eval
        model.eval()
        # validate model per epoch
        valid_loss = 0
        for x,x_lens, y, y_lens, score in valid_generator:
            x, y, score = x.to(device), y.to(device), score.to(device)            
            # output from model
            output = model(x,x_lens,y, y_lens)
            
            # loss computation
            loss = loss_fn(output,score)
            
            # cummulate loss
            valid_loss += loss.detach()

        # report
        if epoch % 10 == 0: 
            print('------------------------------------')
            print(f'epoch-{epoch}:')
            print('------------------------------------')
            print(f' * train loss: {train_loss/len(train_generator)}')
            print(f' * validation loss: {valid_loss/len(valid_generator)}')
        # choose best model
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model = model
    
    # save best model
    torch.save(best_model,f'models/covnet.pt')
    
    return best_model
    

    
def predict(model, sample):

    x, x_lens, y, y_lens  = sample[0], sample[1], sample[2], sample[3]
    # polygon encoder
    x,y = x.to(device), y.to(device)
    model.to(device)
    output = model(x,x_lens,y,y_lens)
    
    return output.detach()

            

def evaluate_prediction(model, test_data):

    # loss 
    mse = nn.L1Loss()

    # num samples
    n_samples = len(test_data)
    
    
    y = [sample[4] for sample in test_data]
    x = [predict(model,sample) for sample in test_data]   
       


    #pdb.set_trace()
    
    x = torch.tensor(x).detach()
    y = torch.tensor(y).detach()
    diff = np.array(np.abs(x.numpy()-y.numpy()))
    # compute loss
    loss = mse(x,y)
    r2 = r2_score(x, y)
    
    # compute r^2 error        
    df = pd.DataFrame(data={'x':x.numpy(),'y':y.numpy(),'diff': diff})
    
            
    plt.scatter(x,y,s=5)
    plt.title(f'samples: {n_samples}, R^2={r2}', loc = 'left')
    plt.xlabel('predicted scores')
    plt.ylabel('true scores')
    

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    img = 'results/results.png'
    plt.savefig(img)
    print(f"Results written to {img}")

    
    print(f'MAE error on test set of size {n_samples}: ',loss.item())
    print(f'R2 score on test set of size {n_samples}: ',r2.item())
    

    return df,img
            
    


if __name__ == "__main__":

        
    parser = argparse.ArgumentParser(description='CovNet')
    parser.add_argument('command', choices = ['train','test','generate'],help='train or test?')
    args = parser.parse_args()
   
    train_args = Params({
        'lr' : 1e-3,
        'num_epochs' : 150,
        'wd' : 1e-5,    
        'batch_size' : 64,
        'sol_sample' : 40    
    })

    model_args = Params({
        'input_size'  : 3,
        'hidden_size' : 50,
    })




    # define instances
    
    # train on sizes up to 200 vertices
    name = '../../dataset/AG/covnet/train/medium'
    data_name = 'medium_data'
    
    # generate new instances
    if args.command == 'generate':        
        print(f'Generating instances with {train_args.sol_sample} sub-optimal solutions')
        instances = generate_instances(name,sol_sample=train_args.sol_sample)
        print(f'Number of instances: {len(instances)}')
        # split data
        data = train_valid_test_split(instances)
        
        
        # serialize        
        with open(f'data/{data_name}_{train_args.sol_sample}.pickle','wb') as f:
            pickle.dump(data, f)
    
    # read train, valid and test    
    # split data
    with open(f'data/{data_name}_{train_args.sol_sample}.pickle','rb') as f:
        data = pickle.load(f)

    # split data to train, valid and test
    train_set, valid_set, test_set = data[0], data[1], data[2]
    
    # model definition
    covnet = CovNet(model_args).to(device)

    # dataloader
    train_generator = DataLoader(train_set, shuffle=False,batch_size=train_args.batch_size,collate_fn=collate_fn_packed)
    valid_generator = DataLoader(valid_set, shuffle=False,batch_size=train_args.batch_size,collate_fn=collate_fn_packed)
    test_generator  = DataLoader(test_set,  shuffle=False,batch_size=1,collate_fn=collate_fn_packed)
    
    # training step
    if args.command == 'train':         
        # train
        model = train(covnet, train_generator, valid_generator, train_args=train_args)                
        evaluate_prediction(model,test_generator)
    
    elif args.command == 'test':        
        model = torch.load(f'models/covnet.pt')
        evaluate_prediction(model,test_generator)
        

    
    

    
    
    
    
    

