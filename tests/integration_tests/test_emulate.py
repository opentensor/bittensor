from __future__ import print_function
import argparse
import bittensor
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class BNet(nn.Module):
    def __init__(self):
        super(BNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, bittensor.__network_dim__)
        self.fc2 = nn.Linear(bittensor.__network_dim__, 10)
        self.wallet = bittensor.wallet()
        self.axon = bittensor.axon(
            wallet = self.wallet,
            port = 8080,
            forward_callback = self.axon_forward,
            backward_callback = self.axon_backward
        )
        self.axon.start()
        self.endpoint = bittensor.Endpoint (
            uid = -1,
            ip = '0.0.0.0',
            ip_type = 4,
            port = 8080,
            modality = 0,
            coldkey = self.wallet.coldkeypub,
            hotkey = self.wallet.hotkey.public_key
        )
        self.dendrite = bittensor.dendrite(
            wallet = self.wallet,
            requires_grad=False
        )

    def show_bt(self):
        print ('cov1', None if self.conv1.weight.grad == None else torch.sum(self.conv1.weight.grad))
        print ('cov2', None if self.conv2.weight.grad == None else torch.sum(self.conv2.weight.grad))
        print ('fc1', None if self.fc1.weight.grad == None else torch.sum(self.fc1.weight.grad))
        print ('fc2', None if self.fc2.weight.grad == None else torch.sum(self.fc2.weight.grad))
        print ('x1', torch.sum(self.bt_x1).item())
        print ('x2', torch.sum(self.bt_x2).item())
        print ('x3', torch.sum(self.bt_x3).item())
        print ('x4', torch.sum(self.bt_x4).item())
        print ('x5', torch.sum(self.bt_x5).item())
        print ('x6', torch.sum(self.bt_x6).item())
        print ('x7', torch.sum(self.bt_x7).item())
        print ('x8', torch.sum(self.bt_x8).item())
        print ('x9', torch.sum(self.bt_x9).item())
        print ('x10', torch.sum(self.bt_x10).item())

    def show_nrm(self):
        print ('cov1', None if self.conv1.weight.grad == None else torch.sum(self.conv1.weight.grad))
        print ('cov2', None if self.conv2.weight.grad == None else torch.sum(self.conv2.weight.grad))
        print ('fc1', None if self.fc1.weight.grad == None else torch.sum(self.fc1.weight.grad))
        print ('fc2', None if self.fc2.weight.grad == None else torch.sum(self.fc2.weight.grad))
        print ('x1', torch.sum(self.nrm_x1).item())
        print ('x2', torch.sum(self.nrm_x2).item())
        print ('x3', torch.sum(self.nrm_x3).item())
        print ('x4', torch.sum(self.nrm_x4).item())
        print ('x5', torch.sum(self.nrm_x5).item())
        print ('x6', torch.sum(self.nrm_x6).item())
        print ('x7', torch.sum(self.nrm_x7).item())
        print ('x8', torch.sum(self.nrm_x8).item())
        print ('x9', torch.sum(self.nrm_x9).item())
        print ('x10', torch.sum(self.nrm_x10).item())

    def axon_backward( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        inputs_x.requires_grad = True
        with torch.enable_grad():
            outputs_y = self.axon_forward( None, inputs_x, None)
            torch.autograd.backward (
                tensors = [outputs_y],
                grad_tensors = [grads_dy]
            )
        return inputs_x.grad

    def axon_forward(self, pubkey: str, inputs: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        self.bt_x0 = torch.squeeze(inputs, dim=1)
        self.bt_x1 = self.conv1( self.bt_x0 )
        self.bt_x2 = F.relu( self.bt_x1 )
        self.bt_x3 = self.conv2( self.bt_x2 )
        self.bt_x4 = F.relu( self.bt_x3 )
        self.bt_x5 = F.max_pool2d( self.bt_x4, 2 )
        self.bt_x6 = self.bt_x5
        self.bt_x7 = torch.flatten( self.bt_x6, 1 )
        self.bt_x8 = self.fc1(self.bt_x7)
        self.bt_x9 = F.relu(self.bt_x8)
        x = self.bt_x9.view( inputs.shape[0], inputs.shape[1], bittensor.__network_dim__ )
        return x

    def bt_forward(self, x):
        x, _ = self.dendrite.forward_image(
            endpoints = self.endpoint,
            inputs = torch.unsqueeze(x, 1)
        )
        x = torch.squeeze(x)
        self.bt_x10 = x
        self.bt_x11 = self.fc2(self.bt_x10)
        self.bt_x12 = F.log_softmax(self.bt_x11, dim=1)
        return self.bt_x12

    def nrm_forward(self, x):
        self.nrm_x0 = x
        self.nrm_x1 = self.conv1(self.nrm_x0)
        self.nrm_x2 = F.relu(self.nrm_x1)
        self.nrm_x3 = self.conv2(self.nrm_x2)
        self.nrm_x4 = F.relu(self.nrm_x3)
        self.nrm_x5 = F.max_pool2d(self.nrm_x4, 2)
        self.nrm_x6 = self.nrm_x5
        self.nrm_x7 = torch.flatten(self.nrm_x6, 1)
        self.nrm_x8 = self.fc1(self.nrm_x7)
        self.nrm_x9 = F.relu(self.nrm_x8)
        self.nrm_x10 = self.nrm_x9
        self.nrm_x11 = self.fc2(self.nrm_x10)
        self.nrm_x12 = F.log_softmax(self.nrm_x11, dim=1)
        return self.nrm_x12

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        if args.bittensor:
            output = model.bt_forward(data)
        else:
            output = model.nrm_forward(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        if args.dry_run:
            if args.bittensor:
                model.show_bt()
            else:
                model.show_nrm()
            break

        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                
    return loss.item()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.bittensor:
                output = model.bt_forward(data)
            else:
                output = model.nrm_forward(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--debug', dest='debug', action='store_true', help='''turn on debug.''', default=False)
    parser.add_argument('--bittensor', dest='bittensor', action='store_true', help='''turn on bittensor training.''', default=False)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Set debug.
    bittensor.logging.set_debug( args.debug )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    start_time = time.time()
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    model = BNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        trloss = train(args, model, device, train_loader, optimizer, epoch)
        tsloss = test(args, model, device, test_loader)
        scheduler.step()
    end_time = time.time()
    print ('training_loss', trloss, '\ttesting_loss', tsloss, '\ttime:', end_time - start_time)

if __name__ == '__main__':
    main()
