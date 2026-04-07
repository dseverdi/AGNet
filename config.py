import os
import time
from utils import save_json, load_json



def set_hyperparameters(hidden_size, batch_size, attention_dim, num_layers, bidirectional, lr, config_path="./data/config/config.json"):
    """
    Set hyperparameters for the model.
    
    Parameters:
    hidden_size (int): Size of the hidden layer.
    batch_size (int): Size of the batch.
    attention_dim (int): Dimension of the attention layer.
    num_layers (int): Number of layers in the model.
    bidirectional (bool): Whether to use a bidirectional model.
    lr (float): Learning rate for the optimizer.
    
    Returns:
    dict: Dictionary containing the hyperparameters.
    """
    
    d = {
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "attention_dim": attention_dim,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "lr": lr
    }
    
    save_json(config_path, d)

def get_hyperparameters(config_path="./data/config/config.json"):
    """
    Get hyperparameters from the config file.
    
    Returns:
    dict: Dictionary containing the hyperparameters.
    """
    
    d = load_json(config_path)
    
    return d

class Hyperparameters:
    def __init__(self, config_path="./data/config/config.json"):
        """
        Initialize the Hyperparameters class by loading values from the config file.
        """
        directory = "/".join(config_path.split("/")[:-1]) 
        filename = config_path.split("/")[-1]
        
        if filename not in os.listdir(directory):
            d = get_hyperparameters()
            save_json(config_path, d)
        
        self.params = get_hyperparameters(config_path)

    def __getitem__(self, key):
        """
        Get a specific hyperparameter by key.
        
        Parameters:
        key (str): The key of the hyperparameter to retrieve.
        
        Returns:
        The value of the specified hyperparameter.
        """
        return self.params.get(key)

    def __repr__(self):
        """
        String representation of the Hyperparameters class.
        
        Returns:
        str: A string representation of the loaded hyperparameters.
        """
        return f"Hyperparameters({self.params})"
    
    def name_str(self):
        """
        Generate a name string based on the hyperparameters.
        
        Returns:
        str: A string representation of the hyperparameters for naming purposes.
        """
        return f"hidden_size_{self['hidden_size']}_batch_size_{self['batch_size']}_attention_dim_{self['attention_dim']}_num_layers_{self['num_layers']}_bidirectional_{self['bidirectional']}_lr_{self['lr']}"

        
    def init_from_name_str(self, name_str, config_path="./data/config/config.json"):
        """
        Set hyperparameters from a name string.
        
        Parameters:
        name_str (str): The name string containing hyperparameter values.
        """
        parts = name_str.split("_")

        self.params = {
            "hidden_size": int(parts[2]),
            "batch_size": int(parts[5]),
            "attention_dim": int(parts[8]),
            "num_layers": int(parts[11]),
            "bidirectional": parts[13] == 'True',
            "lr": float(parts[15])
        }
        save_json(config_path, self.params)
        return self

config_dir = "./data/config/"
os.system(f"mkdir -p {config_dir}")

if __name__ == "__main__":
    0
