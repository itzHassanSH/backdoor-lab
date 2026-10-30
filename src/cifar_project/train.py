#Every nn.Module has a flag, self.training. model.train() sets it to True for the model and all its sub-layers,
# and model.eval() sets it to False. That’s all. Layers can then check that flag in their forward and behave differently

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    model.train()
    # The DataLoader yields batches, not individual samples. Each batch might contain a pair (X, y), where:
    # - X contains the input data for the batch.
    # - y contains the corresponding labels.
    for batch, (train_features, train_labels) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(train_features)
        # calculates loss for all 64 images.
        # Averages the 64 individual losses by default.
        loss = loss_fn(pred, train_labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0 :
                loss, current = loss.item(), batch * batch_size +len(train_features)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    pass



