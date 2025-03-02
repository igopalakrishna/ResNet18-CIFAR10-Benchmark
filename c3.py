import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import torch.profiler

# ============================
#  Function to Measure DataLoader Time (with GPU Sync)
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
    total_time = time.time() - start_time  

    return trainloader, total_time

# ============================
#  Function to Profile DataLoader Performance
# ============================
def train_with_profiler(batch_size=128, num_workers=0):
    print("\n Profiling DataLoader Performance with PyTorch Profiler...\n")

    trainloader, loading_time = get_dataloader(batch_size, num_workers)

    # Initialize PyTorch Profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace_logs"),
        record_shapes=True,
        with_stack=True
    ) as prof:

        for i, (inputs, targets) in enumerate(trainloader):
            if i == 10:  # Limit profiling to first 10 batches
                break
            prof.step()  # Step profiler each batch

    print("\nProfiling Completed. **Download** 'trace.json' and open in Chrome at:")
    print("   chrome://tracing\n")
    return loading_time

# ============================
#  Experiment: Finding Optimal Workers (with GPU Sync)
# ============================
def test_io_performance(batch_size=128, profile=False):
    worker_values = [0, 4, 8, 12, 16, 20, 24]  # Worker counts to test
    prev_time = float('inf')
    best_workers = 0

    print("\n **I/O Performance Test (Dataloader Workers)** \n")
    print(f"{'Workers':<10}{'Data Loading Time (s)':<20}")
    print("=" * 40)

    for workers in worker_values:
        if profile:
            loading_time = train_with_profiler(batch_size, workers)  # Fix: Call profiler function correctly
        else:
            _, loading_time = get_dataloader(batch_size, num_workers=workers)

        print(f"{workers:<10}{loading_time:<20.6f}")

        if loading_time >= prev_time:  # Stop when time no longer decreases
            break
        prev_time = loading_time
        best_workers = workers

    print(f"\n Optimal Number of Workers: {best_workers}\n")
    return best_workers

# ============================
#  Main Function
# ============================
def main(args):
    print("\nRunning Experiment C3: I/O Optimization\n")
    best_workers = test_io_performance(args.batch_size, profile=args.profile)
    print(f"\nBest Number of Workers Found: {best_workers}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler")
    args = parser.parse_args()
    main(args)
