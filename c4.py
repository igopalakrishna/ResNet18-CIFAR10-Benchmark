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
#  ResNet-18 Model Definition
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

# ============================
#  Function to Measure DataLoader & Training Time
# ============================
def get_dataloader(batch_size=128, num_workers=0):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    data_loading_time = time.time() - start_time  

    return trainloader, data_loading_time

# ============================
#  Function to Train Model and Measure Computation Time
# ============================
def train(model, device, trainloader, optimizer, criterion):
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

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

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_time = time.time() - start_time

    return training_time, running_loss / len(trainloader), 100. * correct / total

# ============================
#  Function to Train with PyTorch Profiler
# ============================
def train_with_profiler(batch_size, num_workers, epochs=1):
    print("\n Profiling Training with Profiler...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainloader, data_load_time = get_dataloader(batch_size, num_workers)
    model = ResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Start profiling
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace(f"trace_{num_workers}.json"),
        record_shapes=False,  #  Disabling extra logging for speed
        with_stack=False
    ) as prof:

        # Measure training time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        correct, total, running_loss = 0, 0, 0.0

        for epoch in range(epochs):
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

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                prof.step()  # Step profiler every batch

        # Ensure proper synchronization before measuring time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        training_time = time.time() - start_time

        # Compute final accuracy
        accuracy = 100. * correct / total

    print(f"{num_workers:<10}{data_load_time:<20.6f}{training_time:<20.6f}{(data_load_time + training_time):<20.6f}{accuracy:<10.2f}")



# ============================
#  Experiment: Comparing 1 vs. 4 vs. 8 Workers
# ============================
def compare_workers(batch_size=128, num_epochs=1, profile=False):
    workers_to_test = [1, 4, 8]  # Compare 1 vs. 4 vs. 8 workers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n**Worker Comparison (1 vs. 4 vs. 8 workers)**\n")
    print(f"{'Workers':<10}{'Data Load Time (s)':<20}{'Training Time (s)':<20}{'Total Time (s)':<20}{'Accuracy (%)':<10}")
    print("=" * 80)

    for num_workers in workers_to_test:
        if profile:
            train_with_profiler(batch_size, num_workers, num_epochs)  #  Run profiler
        else:
            trainloader, data_load_time = get_dataloader(batch_size, num_workers)
            model = ResNet18().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

            training_time, avg_loss, accuracy = train(model, device, trainloader, optimizer, criterion)

            total_time = data_load_time + training_time

            print(f"{num_workers:<10}{data_load_time:<20.6f}{training_time:<20.6f}{total_time:<20.6f}{accuracy:<10.2f}")


# ============================
#  Main Function (Accepts Arguments)
# ============================
def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
        parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
        parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler")
        args = parser.parse_args()
    
    compare_workers(args.batch_size, args.epochs, profile=args.profile)

if __name__ == '__main__':
    main()
