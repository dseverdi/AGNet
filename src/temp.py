import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load data
    differences = np.load('differences.npy')
    losses = np.load('losses.npy')

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.plot(differences, label='Differences')
    plt.plot(losses, label='Losses')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Differences and Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('differences_and_losses.png')