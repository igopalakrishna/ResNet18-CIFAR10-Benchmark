import argparse
import time
import sys

# Import all experiment scripts
import c1
import c2
import c3
import c4
import c5
import c6
import c7
import q3  # For Q3 parameter count

# ============================
# Function to Run Experiments with Arguments
# ============================
def run_experiment(experiment, args):
    print(f"\n Running {experiment}...\n")
    start_time = time.time()

    if experiment == "c1":
        c1.main(args)
    elif experiment == "c2":
        c2.main(args)
    elif experiment == "c3":
        c3.main(args)
    elif experiment == "c4":
        c4.main(args)
    elif experiment == "c5":
        c5.main(args)
    elif experiment == "c6":
        c6.main(args)
    elif experiment == "c7":
        c7.main(args)
    elif experiment == "q3":
        q3.main(args)
    else:
        print(f" Unknown experiment: {experiment}")

    elapsed_time = time.time() - start_time
    print(f"\n{experiment} completed in {elapsed_time:.2f} seconds!\n")

# ============================
# Main Function for CLI Execution
# ============================
def main():
    parser = argparse.ArgumentParser(description="Run ML Experiments for ResNet-18 on CIFAR-10")
    parser.add_argument("--exercise", type=str, choices=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "q3"],
                        help="Specify which exercise to run (e.g., c5 for GPU vs CPU)")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all exercises sequentially")
    
    # Common arguments for experiments
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device (cuda or cpu)")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Optimizer")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler")

    args = parser.parse_args()

    if args.run_all:
        print("\n Running ALL Experiments...\n")
        for experiment in ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "q3"]:
            run_experiment(experiment, args)
    elif args.exercise:
        run_experiment(args.exercise, args)
    else:
        parser.print_help()

# ============================
# Run Main if Executed
# ============================
if __name__ == "__main__":
    main()
