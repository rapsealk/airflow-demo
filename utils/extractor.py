import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

root_path = Path(__file__).parent.parent

with open(root_path / "datasets/mnist/train/dataset_info.json", "r") as f:
    dataset_info = json.load(f)

image_size = 28
test_num_examples = dataset_info["splits"]["test"]["num_examples"]

with open(root_path / "datasets/mnist/MNIST/raw/t10k-images-idx3-ubyte", "rb") as f:
    f.read(16)  # Skip the header
    buf = f.read(image_size * image_size * test_num_examples)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(test_num_examples, image_size, image_size)

for i in range(10):
    plt.imsave(root_path / f"datasets/{i}.png", data[i], cmap="gray")
