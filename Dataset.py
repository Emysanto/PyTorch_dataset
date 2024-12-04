import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the dataset and split it among clients
def load_data(num_clients):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    client_datasets = random_split(dataset, [len(dataset)//num_clients]*num_clients)
    return client_datasets

accuracy_history = []
loss_history = []
time_history = []
start_time = time.time()

# Train a single client model
def train(client_model, train_loader, optimizer, criterion, epochs=5):
    client_model.train()
    for epoch in range(epochs):
      for batch_idx, (data, target) in enumerate(train_loader):
          optimizer.zero_grad()
          output = client_model(data)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()

# Aggregate the models' weights
def average_weights(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

# Test the global model
def test(global_model, test_loader):
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = global_model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    accuracy_history.append(correct / len(test_loader.dataset))
    loss_history.append(test_loss)
    time_history.append(time.time() - start_time)

# Main federated learning process
def federated_learning(num_clients, num_rounds):
    client_datasets = load_data(num_clients)
    global_model = Net().to(device)
    client_models = [Net().to(device) for _ in range(num_clients)]

    train_loaders = [DataLoader(client_datasets[i], batch_size=32, shuffle=True) for i in range(num_clients)]
    test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])), batch_size=1000, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    for round in range(num_rounds):
        print('Round ',round+1)
        client_optimizers = [optim.SGD(client_model.parameters(), lr=0.001) for client_model in client_models]
        i=1
        for client_model, train_loader, optimizer in zip(client_models, train_loaders, client_optimizers):
            print('Training client ',i)
            i+=1
            train(client_model, train_loader, optimizer, criterion, epochs=2)
        global_model = average_weights(global_model, client_models)
        test(global_model, test_loader)

        #Update client models
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_clients = 5
    num_rounds = 10
    federated_learning(num_clients, num_rounds)   
