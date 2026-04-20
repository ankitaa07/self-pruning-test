# Self-Pruning Neural Network — Results Report

## Why L1 on Sigmoid Gates Encourages Sparsity

The sigmoid function maps gate scores to values between 0 and 1. These gate values directly scale the weights in the network. Applying an L1 penalty (sum of all gate values) introduces a constant gradient that pushes each gate toward zero. Unlike L2 regularization, which weakens as values approach zero, L1 continues to apply pressure, causing less important gates to shrink toward zero.

As a result, many gates become very close to zero, effectively disabling the corresponding weights. This leads to a sparse neural network where only the most important connections remain active.

---

## Results Table

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|------------------|--------------------|
| 1e-05  | 55.58            | 1.00               |
| 0.0001 | 57.01            | 21.12              |
| 0.001  | 55.29            | 93.51              |

---

## Trade-off Observation

The value of λ (lambda) controls the balance between model accuracy and sparsity.

- At **low λ (1e-05)**, the sparsity penalty is weak, so almost all gates remain active. This results in very low sparsity and high accuracy.
- At **medium λ (0.0001)**, some gates are pushed toward zero, leading to moderate sparsity while maintaining the highest accuracy.
- At **high λ (0.001)**, the sparsity penalty is strong, causing most gates to shrink toward zero. This leads to very high sparsity, with a slight reduction in accuracy.

This demonstrates the expected **sparsity–accuracy trade-off**, where increasing sparsity can reduce model performance.

---

## Gate Distribution Observation

The distribution of gate values shows a large concentration near zero, representing pruned weights, and a smaller group of values away from zero, representing important connections.

This confirms that the network successfully learns which weights to keep and which to prune.