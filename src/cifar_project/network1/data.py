# Dataset, transforms, DataLoader creation
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)


# Testing said Data by visualizing in matplotlab
import matplotlib.pyplot as plt

def plot ():
    figure = plt.figure(figsize=(8, 8))  # interpreted as (8,8, in) where in:= inches
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = int(
            torch.randint(low=0, high=len(training_data), size=(1,)).item())  # returns number for a single value tensor
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(training_data.classes[label])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0), interpolation="nearest")

        # CIFAR-10 (RGB)
        # plt.imshow(img.permute(1, 2, 0))
        # FashionMNIST (grayscale)
        # plt.imshow(img.squeeze(), cmap="gray")

        # plt is a stateful interface that keeps track of the current figure and current Axes (subplot)
    plt.show()



