echo "# ResNet-18 CIFAR-10 Experimentation

## Overview
This repository contains multiple experiments using ResNet-18 on the CIFAR-10 dataset. The experiments compare different optimizers, number of workers, and data loading strategies. The main script to run all experiments is \`lab2.py\`.

## Requirements
Ensure you have the following dependencies installed:
\`\`\`
pip install torch torchvision
\`\`\`

## Running the Experiments
All experiments can be executed using \`lab2.py\` with the \`--exercise\` flag.

### Run a Specific Experiment
To run a specific experiment (e.g., \`c5\`), use:
\`\`\`
python lab2.py --exercise c5
\`\`\`

### Run All Experiments Sequentially
To execute all experiments in sequence:
\`\`\`
python lab2.py --run_all
\`\`\`

### Running on CPU or GPU
By default, the script runs on CUDA if available. You can specify the device explicitly:
\`\`\`
python lab2.py --exercise c5 --device cpu
python lab2.py --exercise c5 --device cuda
\`\`\`

### Profiling with PyTorch Profiler
To enable profiling for an experiment, add the \`--profile\` flag:
\`\`\`
python lab2.py --exercise c5 --profile
\`\`\`
This will generate a \`trace.json\` file that you can view in Chrome by opening:
\`\`\`
chrome://tracing
\`\`\`

### Additional Command-Line Arguments
- \`--epochs\`: Number of epochs to train (default: 5)
- \`--batch_size\`: Batch size for training (default: 128)
- \`--num_workers\`: Number of data loading workers (default: 8)
- \`--lr\`: Learning rate (default: 0.1)
- \`--optimizer\`: Optimizer to use (\`sgd\` or \`adam\`)
\`\`\`
python lab2.py --exercise c5 --epochs 10 --batch_size 64 --num_workers 4 --lr 0.01 --optimizer adam
\`\`\`

## File Structure
- \`lab2.py\` - Main driver script to run all experiments.
- \`c1.py\` - Baseline ResNet-18 experiment.
- \`c2.py\` - Optimized training loop.
- \`c3.py\` - I/O optimization analysis.
- \`c4.py\` - Worker comparison.
- \`c5.py\` - GPU vs CPU performance test.
- \`c6.py\` - Optimizer performance comparison.
- \`c7.py\` - Batch normalization study.
- \`q3.py\` - Model parameter counting.

## Viewing Profiling Results
After running an experiment with \`--profile\`, you can download \`trace.json\` and open it in Chrome:
\`\`\`
chrome://tracing
\`\`\`

## Notes
- Sometimes, the optimal number of workers varies between 4 and 8 due to system scheduling, CPU contention, and parallelization behavior in different environments." > README.md
