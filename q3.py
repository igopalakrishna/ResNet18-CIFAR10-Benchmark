import torch
import torch.nn as nn
import torch.optim as optim

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
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

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
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.fc(out)

# ============================
#  Function to Count Trainable Parameters & Gradients
# ============================
def count_parameters(model, optimizer_name):
    total_params = sum(p.numel() for p in model.parameters())
    total_gradients = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n {optimizer_name} Optimizer - Trainable Parameters: {total_params}")
    print(f" {optimizer_name} Optimizer - Total Gradients: {total_gradients}")

# ============================
#  Experiment Function (Runs from lab2.py)
# ============================
def run_experiment(args):
    device = torch.device(args.device)
    model = ResNet18().to(device)

    print(f"\n Running Parameter Counting Experiment on {args.device.upper()} \n")

    # Compute for SGD Optimizer
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    count_parameters(model, "SGD")

    # Compute for Adam Optimizer
    optimizer_adam = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    count_parameters(model, "Adam")

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
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    args = parser.parse_args()
    main(args)
