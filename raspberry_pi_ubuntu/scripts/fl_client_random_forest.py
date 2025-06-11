import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import flwr as fl
import time
import os
import sys

# Configuração
DATA_PATH = 'intel_lab_data_cleaned.csv'
OUTPUT_DIR = 'ml_results_random_forest'
N_ESTIMATORS = 100
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'
SERVER_ADDRESS = sys.argv[2] if len(sys.argv) > 2 else '192.168.1.100:8080'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Função de avaliação
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'Predictions vs Actual - {model_name} ({CLIENT_ID})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'random_forest_scatter_{CLIENT_ID}.png'))
    plt.close()
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [X.columns[i] for i in indices], rotation=45)
    plt.title(f'Feature Importance - {model_name} ({CLIENT_ID})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'random_forest_importance_{CLIENT_ID}.png'))
    plt.close()
    
    return rmse, mae, r2, y_pred, training_time

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
class RandomForestClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42)
        self.metrics = None

    def get_parameters(self, config):
        return [np.array(v) if isinstance(v, (int, float)) else v for v in self.model.get_params().values()]

    def set_parameters(self, parameters):
        params = self.model.get_params()
        for k, v in zip(params.keys(), parameters):
            params[k] = v
        self.model.set_params(**params)
        return self

    def fit(self, parameters, config):
        start_time = time.time()
        self.set_parameters(parameters)
        rmse, mae, r2, y_pred, train_time = evaluate_model(
            self.model, X_train_scaled, X_test_scaled, y_train, y_test, 'Random Forest'
        )
        self.metrics = {'rmse': rmse, 'mae': mae, 'r2': r2, 'training_time': train_time, 'avg_tree_time': train_time / N_ESTIMATORS}
        return self.get_parameters(config), len(X_train), self.metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return rmse, len(X_test), {'mae': mae, 'r2': r2}

# Iniciar cliente
total_start_time = time.time()
client = RandomForestClient()
fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)
total_time = time.time() - total_start_time
with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
    f.write(f"Total Client Execution Time ({CLIENT_ID}): {total_time:.2f} seconds")