# Art Gallery Problem with Pointer Networks

This repository contains the implementation and training scripts for solving the Art Gallery problem using Pointer Networks and other related models. The workspace is structured to facilitate the development, training, and evaluation of these models.

## Directory Structure

- `AGMnet/`: Main directory for source code and related files.
  - `dataset/`: Contains scripts and data for generating and handling datasets.
  - `train.py`: Script for training the Pointer Network model.
  - `retrain.py`: Script for retraining the model with additional configurations.
  - `CovNet.ipynb`: Jupyter notebook for experimenting with the CovNet model.
  - `covnet.py`: Python script implementing the CovNet model.
- `README.md`: Documentation file describing the project.

## Key Scripts and Notebooks

- `train.py`: Script for training the Pointer Network model with various configurations.
- `retrain.py`: Script for retraining the model with additional configurations and parameters.
- `CovNet.ipynb`: Jupyter notebook for experimenting with the CovNet model.
- `covnet.py`: Python script implementing the CovNet model, including data loading, training, and evaluation functions.

## Datasets

The dataset directory (`dataset/AG/development/`) contains various complexity levels of terrain data used for training and validation. The datasets are divided into `train`, `dev`, and `test` sets, each containing `.pol` files.

## Models

- `PtrNet`: Pointer Network model used for solving the Art Gallery problem.
- `CovNet`: Another neural network model implemented in the workspace.

## Configuration and Parameters

The training scripts (`train.py`, `retrain.py`) use `argparse` to handle various command-line arguments for configuring the training process, such as batch size, learning rate, number of epochs, and more.

## Example Usage

Here is an example of how to load and display the dataset information using a custom function:

```python
dataset_dir = './dataset/AG/development/'
samples = {s : [f for f in os.listdir(dataset_dir+s) if f.endswith('.pol')] for s in ['train','dev','test']}
df = pd.DataFrame.from_dict(samples, orient='index').transpose()
# Custom function to count the number of samples in each dataset
df.loc['total'] = df.count()
df.loc['total'].sum()
df.loc['ratio'] = df.loc['total'] / df.loc['total'].sum()
df.loc['total'].sum()

# Display the dataframe
df.head()