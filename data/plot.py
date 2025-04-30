import numpy as np
import csv
import matplotlib.pyplot as plt

with open('./mnist_test.csv', 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    next(csvreader)
    for i, data in enumerate(csvreader):
        if i > 10:
            break
        # The first column is the label
        label = data[0]

        # The rest of columns are pixels
        pixels = data[1:]

        # Convert to NumPy array and reshape
        pixels = np.array(pixels, dtype='int64').reshape((28, 28))

        # Plot and save
        plt.imshow(pixels, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')  # optional: turn off axis
        plt.savefig(f'mnist_img_{i}_label_{label}.png', bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to avoid memory issues
