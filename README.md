# Self-Pruning Neural Network (CIFAR-10)

This project implements a self-pruning neural network using a custom `PrunableLinear` layer in PyTorch. The model learns to automatically prune unnecessary weights using learnable gates and L1 regularization.

##  Key Features
- Custom prunable linear layer (no use of torch.nn.Linear)
- Learnable gate scores with sigmoid activation
- L1 sparsity regularization
- Demonstrates sparsity vs accuracy trade-off

##  Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 1e-05  | 55.58       | 1.00         |
| 0.0001 | 57.01       | 21.12        |
| 0.001  | 55.29       | 93.51        |

##  Observations
- Increasing λ increases sparsity
- High sparsity slightly reduces accuracy
- Medium λ gives best trade-off

##  How to Run

```bash
python self_pruning_cifar10.py
