#!/usr/bin/env python3
import time
import torch
from torchmps import MPS
from torchvision import transforms, datasets

# Miscellaneous initialization
torch.manual_seed(0)
start_time = time.time()

# MPS parameters
bond_dim = 20
adaptive_mode = False
periodic_bc = False

# Training parameters
num_train = 2000
num_test = 1000
batch_size = 100
num_epochs = 20
learn_rate = 1e-4
l2_reg = 0.0

# Initialize the MPS module
mps = MPS(
    input_dim=28 ** 2,
    output_dim=10,
    bond_dim=bond_dim,
    adaptive_mode=adaptive_mode,
    periodic_bc=periodic_bc,
)

# Set our loss function and optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)

# Get the training and test sets
transform = transforms.ToTensor()
train_set = datasets.MNIST("./mnist", download=True, transform=transform)
test_set = datasets.MNIST("./mnist", download=True, transform=transform, train=False)

# Put MNIST data into dataloaders
samplers = {
    "train": torch.utils.data.SubsetRandomSampler(range(num_train)),
    "test": torch.utils.data.SubsetRandomSampler(range(num_test)),
}
loaders = {
    name: torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=samplers[name], drop_last=True
    )
    for (name, dataset) in [("train", train_set), ("test", test_set)]
}
num_batches = {
    name: total_num // batch_size
    for (name, total_num) in [("train", num_train), ("test", num_test)]
}

print(
    f"Training on {num_train} MNIST images \n"
    f"(testing on {num_test}) for {num_epochs} epochs"
)
print(f"Maximum MPS bond dimension = {bond_dim}")
print(f" * {'Adaptive' if adaptive_mode else 'Fixed'} bond dimensions")
print(f" * {'Periodic' if periodic_bc else 'Open'} boundary conditions")
print(f"Using Adam w/ learning rate = {learn_rate:.1e}")
if l2_reg > 0:
    print(f" * L2 regularization = {l2_reg:.2e}")
print()

# Let's start training!
for epoch_num in range(1, num_epochs + 1):
    running_loss = 0.0
    running_acc = 0.0

    for inputs, labels in loaders["train"]:
        inputs, labels = inputs.view([batch_size, 28 ** 2]), labels.data

        # Call our MPS to get logit scores and predictions
        scores = mps(inputs)
        _, preds = torch.max(scores, 1)

        # Compute the loss and accuracy, add them to the running totals
        loss = loss_fun(scores, labels)
        with torch.no_grad():
            accuracy = torch.sum(preds == labels).item() / batch_size
            running_loss += loss
            running_acc += accuracy

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"### Epoch {epoch_num} ###")
    print(f"Average loss:           {running_loss / num_batches['train']:.4f}")
    print(f"Average train accuracy: {running_acc / num_batches['train']:.4f}")

    # Evaluate accuracy of MPS classifier on the test set
    with torch.no_grad():
        running_acc = 0.0

        for inputs, labels in loaders["test"]:
            inputs, labels = inputs.view([batch_size, 28 ** 2]), labels.data

            # Call our MPS to get logit scores and predictions
            scores = mps(inputs)
            _, preds = torch.max(scores, 1)
            running_acc += torch.sum(preds == labels).item() / batch_size

    print(f"Test accuracy:          {running_acc / num_batches['test']:.4f}")
    print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")
