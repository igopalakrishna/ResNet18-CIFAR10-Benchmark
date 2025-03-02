import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
import time

import torch.profiler

# ============================
# ResNet-18 Model Definition
# ============================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.fc(out)

# ============================================
# DataLoader Function (with Required Transforms)
# ============================================
def get_dataloader(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader

# ============================
# Training Function
# ============================
def train(model, device, trainloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        torch.cuda.synchronize()  # Ensure accurate GPU timing
        start_time = time.time()

        running_loss = 0.0
        correct = 0
        total = 0
        
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
        
        torch.cuda.synchronize()  # Ensure all GPU operations are completed
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f} - Accuracy: {100. * correct / total:.2f}% - Time: {epoch_time:.4f}s")


# ============================
# Function to Train with PyTorch Profiler (Chrome Tracing)
# ============================
def train_with_profiler(model, device, trainloader, optimizer, criterion, epochs=5):
    model.train()

    print("\n Profiling Enabled. Generating Chrome Trace Log...\n")

    # Initialize the PyTorch Profiler for Chrome Trace
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        with_stack=True
    )
    prof.start()

    for epoch in range(epochs):
        torch.cuda.synchronize()
        start_time = time.time()

        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            with torch.profiler.record_function("forward_pass"):
                outputs = model(inputs)
            with torch.profiler.record_function("loss_calculation"):
                loss = criterion(outputs, targets)
            with torch.profiler.record_function("backward_pass"):
                loss.backward()
            with torch.profiler.record_function("optimizer_step"):
                optimizer.step()

            prof.step()  # Step profiler each iteration

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        accuracy = 100. * correct / total

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f} - Accuracy: {accuracy:.2f}% - Time: {epoch_time:.4f}s")

    prof.stop()
    
    #  Export trace for Chrome Tracing
    prof.export_chrome_trace("trace.json")
    print("\nProfiling Completed. **Download** 'trace.json' and open in Chrome at:")
    print("   chrome://tracing\n")

    return running_loss / epochs, accuracy, epoch_time


# ========================================
# Function to Count Trainable Parameters
# ========================================
def count_params(model, optimizer_name):
    total_params = sum(p.numel() for p in model.parameters())
    total_gradients = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{optimizer_name} Optimizer - Total Trainable Parameters: {total_params}")
    print(f"{optimizer_name} Optimizer - Total Gradients: {total_gradients}")

# ============================
# Main Function
# ============================
def main(args=None):  # Accepts args from lab2.py
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
        parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
        parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
        parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer type')
        parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler")
        args = parser.parse_args()

    device = torch.device(args.device)
    trainloader = get_dataloader(args.batch_size, args.num_workers)
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    if args.profile:
        train_with_profiler(model, device, trainloader, optimizer, criterion, args.epochs)
    else:
        train(model, device, trainloader, optimizer, criterion, args.epochs)

    count_params(model, args.optimizer)

# Run the main function when script is executed
if __name__ == '__main__':
    main()
