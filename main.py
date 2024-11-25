import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
import numpy as np
data = pd.read_csv('/content/fegnn.csv', sep=';')
data = data.replace(',', '.', regex=True).astype(float)
X = data.iloc[:, :-1].values  # Признаки
y = data.iloc[:, -1].values   # Метки
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
num_models = 10
input_size = X.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = len(np.unique(y))
models = [SimpleNet(input_size, hidden_size1, hidden_size2, output_size) for _ in range(num_models)]
def train_model(model, train_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
def train_ensemble(models, dataset, sample_size=0.8, num_epochs=20):
    for model in models:
        sample_indices = torch.randperm(len(dataset))[:int(sample_size * len(dataset))]
        subset = Subset(dataset, sample_indices)
        train_loader = DataLoader(subset, batch_size=16, shuffle=True)
        train_model(model, train_loader, num_epochs)
def ensemble_predict(models, data):
    outputs = [model(data) for model in models]
    avg_output = torch.mean(torch.stack(outputs), dim=0)
    _, predicted = torch.max(avg_output, 1)
    return predicted
def calculate_accuracy(models, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            predictions = ensemble_predict(models, data)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total * 100
    return accuracy
train_ensemble(models, dataset)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
accuracy = calculate_accuracy(models, data_loader)
print(f'Точность ансамбля: {accuracy:.2f}%')

