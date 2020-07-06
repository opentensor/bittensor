import opentensor

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def main():
    n_epochs = 3
    batch_size_train = 64
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    # Dataset.
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root="~/tmp/", train=True, download=True,
            transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
            batch_size=batch_size_train, shuffle=True)


    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
   

    # opentensor Metagraph
    identity = opentensor.Identity()
    metagraph = opentensor.Metagraph(identity)

    # Keys object.
    # projects from/to opentensor_pb2.Node to a variable sized tensor
    key_dim = 100
    keymap = opentensor.Keys(key_dim)

    # Gate: object for a trained lookup of keys
    topk = 10 # number of keys to choose for each examples.
    n_keys = 10  
    x_dim = 784 # key rank input size to the network.
    gate = opentensor.Gate(x_dim, topk, key_dim)
    
    # Object for dispatching / combining gated inputs
    dispatcher = opentensor.Dispatcher()

    # Node to serve on metagraph.
    class Mnist(opentensor.Node): 
        def fwd (self, key, tensor):
            return network(data)

        def bwd (self, key, tensor):
            pass

    # Subscribe the model encoder to the graph.
    metagraph.subscribe(Mnist())

  
    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
    
            output = network(data)
    
            loss = F.nll_loss(output, target)
            loss.backward()
            
            optimizer.step()
    
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


    for epoch in range(1, n_epochs + 1):
        train(epoch)



if __name__ == "__main__":
    main()
