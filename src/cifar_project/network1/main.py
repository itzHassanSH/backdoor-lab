import os
import torch
from torch import nn
from torch.utils.data import DataLoader

# import the training- and test-dataset from data
from data import training_data, test_data

from cifar_project.network1.model import NeuralNetwork

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

from train import train_loop, test_loop

MODEL_PATH = "cifar_model.pth"
def train_and_test(best_loss=float("inf")):
    learning_rate = 1e-3
    batch_size = 64
    epochs = 20

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

     # we try and prevent overfitting -> prevent updating model if loss becomes worse after an epoch
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(training_dataloader, model, loss_fn, optimizer, batch_size)
        test_loss = test_loop(testing_dataloader, model, loss_fn)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), MODEL_PATH)
    print(f"Done! Best test loss {best_loss:>8f}, weights in {MODEL_PATH}")
    load_model()

def load_model():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()


def use():
    # Using Model (wrong code so far!)
    # CIFAR10 images are (3, 32, 32), with no size-1 dimension, so img stays (3, 32, 32).
    model.eval()
    with torch.no_grad():
        test_features, test_labels = next(iter(testing_dataloader))
        logits = model(test_features.to(device))  # model lives on device, thus img must be moved from CPU
        pred_probability = nn.Softmax(dim=1)(logits)
        # argmax(1) returns the position (index) of the highest value - 1 is the dimension
        y_pred = pred_probability.argmax(1)
    y_pred = y_pred.cpu()
    for i in range(0, 64):
        print(f"Predicted: {y_pred[i].item()}, actual: {test_labels[i].item()}")

def plot ():
    figure = plt.figure(figsize=(8, 8))
    col, rows = 8, 8

if __name__ == "__main__":
    best_loss_outer = float("inf")
    if os.path.exists(MODEL_PATH):
        load_model()          # skip training, reuse saved progress
        best_loss_outer = test_loop(testing_dataloader, model, loss_fn=nn.CrossEntropyLoss())
    train_and_test(best_loss_outer)      # train once, saves at the end
    use()




