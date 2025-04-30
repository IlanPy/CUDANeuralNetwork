import numpy as np
import pandas as pd
from medmnist import OCTMNIST

ds = OCTMNIST(split='test', download=True, size=64)
images = ds.imgs  # shape: (n, 224, 224)
labels = ds.labels  # shape: (n, 1) or (n,) depending on task

# Flatten and combine
flat_images = images.reshape(images.shape[0], -1)
data = np.concatenate([labels, flat_images], axis=1)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("octmnist_test_64.csv", index=False)

