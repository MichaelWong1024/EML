from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()
        # Block1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # block2 part1
        self.conv2_1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_1_1 = nn.BatchNorm2d(64)
        self.conv2_1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_1_2 = nn.BatchNorm2d(64)
        self.dropout2_1 = nn.Dropout(0.5)
        # block2 identity
        self.conv2_2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_2_1 = nn.BatchNorm2d(64)
        self.conv2_2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_2_2 = nn.BatchNorm2d(64)
        self.dropout2_2 = nn.Dropout(0.5)
        # block3 part1
        self.conv3_1_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_1_2 = nn.BatchNorm2d(128)
        self.conv3_1_3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        self.dropout3_1 = nn.Dropout(0.5)
        # block3 identity
        self.conv3_2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_2_1 = nn.BatchNorm2d(128)
        self.conv3_2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_2_2 = nn.BatchNorm2d(128)
        self.dropout3_2 = nn.Dropout(0.5)
        # block4 part1
        self.conv4_1_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4_1_1 = nn.BatchNorm2d(256)
        self.conv4_1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_1_2 = nn.BatchNorm2d(256)
        self.conv4_1_3 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        self.dropout4_1 = nn.Dropout(0.5)
        # block4 identity
        self.conv4_2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_2_1 = nn.BatchNorm2d(256)
        self.conv4_2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_2_2 = nn.BatchNorm2d(256)
        self.dropout4_2 = nn.Dropout(0.5)
        # block5 part1
        self.conv5_1_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5_1_1 = nn.BatchNorm2d(512)
        self.conv5_1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_1_2 = nn.BatchNorm2d(512)
        self.conv5_1_3 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)
        self.dropout5_1 = nn.Dropout(0.5)
        # block5 identity
        self.conv5_2_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_2_1 = nn.BatchNorm2d(512)
        self.conv5_2_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_2_2 = nn.BatchNorm2d(512)
        self.dropout5_2 = nn.Dropout(0.5)
        # fully connected
        self.average = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # block1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x1 = self.maxpool(x)
        # block2 part1
        x = self.conv2_1_1(x1)
        x = self.bn2_1_1(x)
        x = F.relu(x)
        x = self.conv2_1_2(x)
        x = self.bn2_1_2(x)
        x = self.dropout2_1(x)
        x2_1 = F.relu(x1 + x)
        # block2 identity
        x = self.conv2_2_1(x2_1)
        x = self.bn2_2_1(x)
        x = F.relu(x)
        x = self.conv2_2_2(x)
        x = self.bn2_2_2(x)
        x = self.dropout2_2(x)
        x2_2 = F.relu(x2_1 + x)
        # block3 part1
        x = self.conv3_1_1(x2_2)
        x = self.bn3_1_1(x)
        x = F.relu(x)
        x = self.conv3_1_2(x)
        x = self.bn3_1_2(x)
        x = self.dropout3_1(x)
        x3_1 = F.relu(self.conv3_1_3(x2_2) + x)
        # block3 identity
        x = self.conv3_2_1(x3_1)
        x = self.bn3_2_1(x)
        x = F.relu(x)
        x = self.conv3_2_2(x)
        x = self.bn3_2_2(x)
        x = self.dropout3_2(x)
        x3_2 = F.relu(x3_1 + x)
        # block4 part1
        x = self.conv4_1_1(x3_2)
        x = self.bn4_1_1(x)
        x = F.relu(x)
        x = self.conv4_1_2(x)
        x = self.bn4_1_2(x)
        x = self.dropout4_1(x)
        x4_1 = F.relu(self.conv4_1_3(x3_2) + x)
        # block4 identity
        x = self.conv4_2_1(x4_1)
        x = self.bn4_2_1(x)
        x = F.relu(x)
        x = self.conv4_2_2(x)
        x = self.bn4_2_2(x)
        x = self.dropout4_2(x)
        x4_2 = F.relu(x4_1 + x)
        # block5 part1
        x = self.conv5_1_1(x4_2)
        x = self.bn5_1_1(x)
        x = F.relu(x)
        x = self.conv5_1_2(x)
        x = self.bn5_1_2(x)
        x = self.dropout5_1(x)
        x5_1 = F.relu(self.conv5_1_3(x4_2) + x)
        # block5 identity
        x = self.conv5_2_1(x5_1)
        x = self.bn5_2_1(x)
        x = F.relu(x)
        x = self.conv5_2_2(x)
        x = self.bn5_2_2(x)
        x = self.dropout5_2(x)
        x5_2 = F.relu(x5_1 + x)
        # fully connected
        x = self.average(x5_2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        output = self.fc2(x)

        return output


def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output[:3], target[:3])
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
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
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Res18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        test_loss = test(model, device, test_loader)
        scheduler.step(metrics=test_loss)

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_cnn.pt")


if __name__ == '__main__':
    main()
