import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

# ============================
#  ResNet-18 Model WITHOUT BatchNorm
# ============================
class BasicBlockNoBN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18NoBN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18NoBN, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlockNoBN(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlockNoBN(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.fc(out)

# ============================
#  DataLoader Function
# ============================
def get_dataloader(batch_size=128, num_workers=8):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader

# ============================
#  Training Function
# ============================
def train(model, device, trainloader, optimizer, criterion, epochs=5):
    model.train()
    total_loss, total_accuracy, total_time = 0, 0, 0

    for epoch in range(epochs):
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()

        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        torch.cuda.synchronize() if device.type == "cuda" else None
        epoch_time = time.time() - start_time
        accuracy = 100. * correct / total

        total_loss += running_loss / len(trainloader)
        total_accuracy += accuracy
        total_time += epoch_time

        print(f"Epoch {epoch+1}/5 - Loss: {running_loss/len(trainloader):.4f} - Accuracy: {accuracy:.2f}% - Time: {epoch_time:.4f}s")

    avg_loss = total_loss / epochs
    avg_accuracy = total_accuracy / epochs
    avg_time = total_time / epochs

    print(f"\n **Final Summary Without BatchNorm**")
    print(f" Average Loss: {avg_loss:.4f}")
    print(f" Average Accuracy: {avg_accuracy:.2f}%")
    print(f" Average Training Time per Epoch: {avg_time:.4f} seconds\n")

# ============================
#  Experiment Function (Runs from lab2.py)
# ============================
def run_experiment(args):
    device = torch.device(args.device)
    trainloader = get_dataloader(args.batch_size)

    print(f"\n Training Without Batch Normalization Using SGD on {args.device.upper()} \n")
    model = ResNet18NoBN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    train(model, device, trainloader, optimizer, criterion, args.epochs)

# ============================
#  Main Function
# ============================
def main(args):
    run_experiment(args)

# ============================
#  Run When Called from CLI
# ============================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda"], help="Device (GPU only)")
    args = parser.parse_args()
    main(args)
