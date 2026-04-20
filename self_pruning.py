import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ======================================
# 1. PRUNABLE LINEAR LAYER
# ======================================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.kaiming_uniform_(self.weight, a=0.01)

        # Better initialization for pruning
        self.gate_scores = nn.Parameter(torch.randn_like(self.weight) * 0.1)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)

        # Soft pruning during training
        effective_weight = self.weight * gates

        return F.linear(x, effective_weight, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)


# ======================================
# 2. MODEL (DEEP + SAFE STRUCTURE)
# ======================================
class HybridPruningNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            PrunableLinear(3072, 800),
            PrunableLinear(800, 400),
            PrunableLinear(400, 200),
            PrunableLinear(200, 10)
        ])

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))

        x = self.layers[-1](x)
        return x

    def get_all_gates(self):
        all_gates = []
        for layer in self.layers:
            all_gates.append(layer.get_gates().reshape(-1))
        return torch.cat(all_gates)


# ======================================
# 3. SPARSITY LOSS (TRUE L1)
# ======================================
def sparsity_loss(model):
    total = 0
    for layer in model.layers:
        total += layer.get_gates().sum()
    return total


# ======================================
# 4. DATA LOADING
# ======================================
def get_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader


# ======================================
# 5. TRAINING
# ======================================
def train_model(lambda_val, train_loader, device, epochs=20):
    model = HybridPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining with lambda = {lambda_val}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            ce_loss = F.cross_entropy(outputs, labels)
            sp_loss = sparsity_loss(model)

            loss = ce_loss + lambda_val * sp_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {100*correct/total:.2f}%")

    return model


# ======================================
# 6. EVALUATION
# ======================================
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()

    return 100 * correct / len(test_loader.dataset)


# ======================================
# 7. SPARSITY METRIC
# ======================================
def calculate_sparsity(model, threshold=1e-2):
    gates = model.get_all_gates()
    total = gates.numel()
    pruned = (gates < threshold).sum().item()

    return 100 * pruned / total


# ======================================
# 8. PLOT
# ======================================
def plot_distribution(model):
    gates = model.get_all_gates().detach().cpu().numpy()

    plt.figure(figsize=(6,4))
    plt.hist(gates, bins=60)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()


# ======================================
# 9. MAIN
# ======================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, test_loader = get_data()

    lambdas = [1e-5, 1e-4, 1e-3]

    results = []
    best_model = None
    best_acc = 0

    for lam in lambdas:
        model = train_model(lam, train_loader, device, epochs=20)

        acc = evaluate(model, test_loader, device)
        sparsity = calculate_sparsity(model)

        print(f"\nLambda: {lam}")
        print(f"Test Accuracy: {acc:.2f}%")
        print(f"Sparsity: {sparsity:.2f}%")

        results.append((lam, acc, sparsity))

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print("\nFINAL RESULTS:")
    for r in results:
        print(f"Lambda: {r[0]} | Accuracy: {r[1]:.2f}% | Sparsity: {r[2]:.2f}%")

    plot_distribution(best_model)