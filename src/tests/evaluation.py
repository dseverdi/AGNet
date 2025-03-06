import numpy as np
import pandas as pd
import os, sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# imports
from evaluate import AGNetSearch, Opt
from demo import VisSample, evaluate_polygon_visibility



# evaluation interface
class Evaluator:
    def __init__(self, solver, evaluate = evaluate_polygon_visibility):
        self.solver = solver        
        self.evaluate = evaluate
        self.results = {}

    def evaluate_metrics(self, test : dict) -> dict:
        """
        Evaluate metrics on test dataset
        """

        # read test data
        key, data, dataset_dir, n = test['name'], test['data'], test['dataset_dir'], len(test['data'])        

        # initialize arrays
        _coverages, _ratios = np.zeros(n), np.zeros(n)

        for i, instance in enumerate(data):
            # read sample
            print(f'Processing {instance}                         ', end='\r')
            _instance = os.path.join(dataset_dir, instance)
            sample = VisSample.read_samples(path=_instance, sol_sample=1)[0]           

            # prediction        
            solution = self.solver.predict(_instance, beam_width=4)[0]

            # compute opt and coverage
            _, _, _, opt, coverage = self.evaluate(sample, solution)
            # accumulate
            _coverages[i] = coverage
            _ratios[i] = len(solution) / len(opt)

        self.results[key] = {
            'counts': len(_coverages),
            'mean coverage': np.mean(_coverages),
            'std coverage': np.std(_coverages),
            'mean ratio': np.mean(_ratios),
            'std ratio': np.std(_ratios)
        }

        return self.results

    def stats(self, testbed):
        """
        Run evaluation on testbed
        """

        # list files
        df = pd.DataFrame([file for file in os.listdir(testbed['dataset_dir']) if file.endswith('.pol')], columns=['name'])

        # check patterns
        for i, re in enumerate(testbed['regex']):
            print(f"\nPattern {i + 1}/{len(testbed['regex'])}:")

            # read data using regex
            data = df[df.name.str.match(re)].values.squeeze().tolist()

            # define rows
            test = {'name': re, 'data': data, 'dataset_dir': testbed['dataset_dir']}

            # evaluate metrics
            self.results = self.evaluate_metrics(test)

        # show as dataframe
        report = pd.DataFrame.from_dict(self.results, orient='index')
        # Create results directory if it doesn't exist
        os.makedirs('./results', exist_ok=True)
        
        # Save report to JSON file
        report.to_json(f"./results/{testbed['name']}.json")

        display(report)
        
        return report

    def boxplot(self, testbed):
        """
        Run evaluation and generate boxplot
        """
        # list all instances
        files = [file for file in os.listdir(testbed['dataset_dir']) if file.endswith('.pol')]
        # create df for results
        data = pd.DataFrame(columns=['name', 'type', 'coverage', 'ratio'])
        # filter out files via regex
        for regex in testbed['regex']:
            p = re.compile(regex)
            for file in files:
                if p.match(file):  # if file matches regex
                    test = {'name': regex, 'data': [file], 'dataset_dir' : testbed['dataset_dir']}  # wrap data to use API
                    results = self.evaluate_metrics(test)
                    cov, r = results[regex]['mean coverage'], results[regex]['mean ratio']  # read mean for singletons
                    data.loc[len(data)] = [file, regex, cov, r]  # set instance name, type, coverage and ratio

        # Determine unique values in the 'type' column
        unique_types = data['type'].unique()
        labels = unique_types.tolist()  # Convert unique types to a list
        colors = sns.color_palette("husl", len(labels))  # Generate a color palette with enough colors

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first boxplot with different colors
        sns.boxplot(ax=axes[0], data=data, x="type", y="coverage", hue="type", palette=colors, legend=False)
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels)

        # Plot the second boxplot with different colors
        sns.boxplot(ax=axes[1], data=data, x="type", y="ratio", hue="type", palette=colors, legend=False)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels)

        plt.tight_layout()
        plt.show()

        return data



if __name__ == '__main__':
    results = {}

    dataset_dir = './dataset/development/test'

    testbeds = [{ 
        'name' : 'AGNet_testbed_1',
        'dataset_dir' : dataset_dir,
        'regex' : [
            '.*?-[1-5]\d-', # 1-5 tens size
            '.*?-[6-9]\d-', # 6-9 tens sizes,
            '.*?-1\d\d-', # 100s size  
        ] 
    },

    { 
        'name' : 'AGNet_testbed_2',    
        'dataset_dir' : dataset_dir,
        'regex' : [
            '.*?-2\d\d-', # 200s
            '.*?-3\d\d-', # 300s
            '.*?-4\d\d-', # 400s  
            '.*?-5\d\d-', # 500s  
        ] 
    }
    ]

    # solvers    
    # load model
    m1  = './models/supervised/trained_models/ag_clusters_numsols-1_ne-100_bs-64_hs-256_tfr-0.5_wd-1e-05_lr-0.001_bidirectional_normalized.pt' 
    #opt value
    opt_solver = Opt()
    # PTrNet search
    agnet_solver  = AGNetSearch(m1)

    # Evaluate model
    evaluator = Evaluator(agnet_solver)
    _ = evaluator.stats(testbeds[0])