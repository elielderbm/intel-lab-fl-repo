import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import sys

# Configuração
DATA_PATH = 'intel_lab_data_cleaned.csv'
OUTPUT_DIR = 'ml_results_cnn'
N_EPOCHS = 100
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'
SERVER_ADDRESS = sys.argv[2] if len(sys.argv) > 2 else '192.168.1.100:8080'
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definir CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 1, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adicionar canal
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Função de avaliação
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    start_time = time.time()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                  torch.tensor(y_train.values, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    epoch_times = []
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        epoch_times.append(time.time() - epoch_start)
    
    training_time = time.time() - start_time
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).squeeze().cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'Predictions vs Actual - {model_name} ({CLIENT_ID})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'cnn_scatter_{CLIENT_ID}.png'))
    plt.close()
    
    return rmse, mae, r2, y_pred, training_time, epoch_times

# Carregar e dividir dados
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATA_PATH}' não encontrado.")
    exit(1)

np.random.seed(42)
client_data = df.sample(frac=1.0 / 3.0, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)
X = client_data[['moteid', 'humidity', 'light', 'voltage']]
y = client_data['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cliente Flower
class CNNClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = CNN().to(device)
        self.metrics = None

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v).to(device)
        self.model.load_state_dict(state_dict)
        return self

    def fit(self, parameters, config):
        start_time = time.time()
        self.set_parameters(parameters)
        rmse, mae, r2, y_pred, train_time, epoch_times = evaluate_model(
            self.model, X_train_scaled, X_test_scaled, y_train, y_test, 'CNN'
        )
        self.metrics = {'rmse': rmse, 'mae': mae, 'r2': r2, 'training_time': train_time, 'epoch_times': epoch_times}
        return self.get_parameters(config), len(X_train), self.metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            y_pred = self.model(X_test_tensor).squeeze().cpu().numpy()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return rmse, len(X_test), {'mae': mae, 'r2': r2}

# Iniciar cliente
total_start_time = time.time()
client = CNNClient()
fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)
total_time = time.time() - total_start_time
with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
    f.write(f"Total Client Execution Time ({CLIENT_ID}): {total_time:.2f} seconds")