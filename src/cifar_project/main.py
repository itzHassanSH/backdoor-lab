import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# import the training- and test-dataset from data
from data import training_data, test_data

from cifar_project.model import NeuralNetwork

# for plotting each image that we predict for
import matplotlib.pyplot as plt

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)

training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
testing_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# training:
# We define the following hyperparameters for training:
## - Number of Epochs - the number of times to iterate over the dataset
## - Batch Size - the number of data samples propagated through the network before the parameters are updated
## - Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed,
## while large values may result in unpredictable behavior during training.

learning_rate = 1e-3
batch_size = 64
epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



# Using Model (wrong code so far!)
train_features, train_labels = next(iter(training_dataloader))
for i in range(0, 64):
    # CIFAR10 images are (3, 32, 32), with no size-1 dimension, so img stays (3, 32, 32).
    img = train_features[i].unsqueeze(0)
    label = train_labels[i]

    logits = model(img)
    pred_probability = nn.Softmax(dim=1)(logits)
    # argmax(1) returns the position (index) of the highest value - 1 is the dimension
    y_pred = pred_probability.argmax(1)
    print(f"Predicted class: {y_pred}")

def plot ():
    figure = plt.figure(figsize=(8, 8))
    col, rows = 8, 8




