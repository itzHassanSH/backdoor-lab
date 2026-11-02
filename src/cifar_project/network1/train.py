#Every nn.Module has a flag, self.training. model.train() sets it to True for the model and all its sub-layers,
# and model.eval() sets it to False. That’s all. Layers can then check that flag in their forward and behave differently
import torch


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    device = next(model.parameters()).device  # wherever the model lives
    model.train()
    # The DataLoader yields batches, not individual samples. Each batch might contain a pair (X, y), where:
    # - X contains the input data for the batch
    # - y contains the corresponding labels
    for batch, (train_features, train_labels) in enumerate(dataloader):
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        # compute prediction and loss
        pred = model(train_features)
        # calculates loss for all 64 images
        # Averages the 64 individual losses by default
        loss = loss_fn(pred, train_labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # The code prints progress every 100 batches
        if batch % 100 == 0 :
                loss, current = loss.item(), batch * batch_size +len(train_features)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    device = next(model.parameters()).device

    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss



