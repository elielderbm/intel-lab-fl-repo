import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Configuração
OUTPUT_DIR = 'ml_results_linear_regression'
N_ROUNDS = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estratégia FedAvg personalizada
class LinearRegressionStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.global_metrics = []
        self.client_metrics = []
        self.weight_variances = []
        self.communication_times = []
        self.aggregation_times = []
        self.round_start_time = None

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start_time = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        start_time = time.time()
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        aggregation_time = time.time() - start_time
        self.aggregation_times.append({'round': server_round, 'time': aggregation_time})
        metrics = [res.metrics for _, res in results]
        client_ids = [res.client.cid for _, res in results]
        self.client_metrics.append({
            'round': server_round,
            'clients': {cid: {'rmse': m['rmse'], 'mae': m['mae'], 'r2': m['r2'], 'training_time': m['training_time']} 
                       for cid, m in zip(client_ids, metrics)}
        })
        if aggregated_parameters:
            weights = [np.concatenate([np.array(p).flatten() for p in res.parameters]) for _, res in results]
            variance = np.var(weights, axis=0).mean() if weights else 0.0
            self.weight_variances.append({'round': server_round, 'variance': variance})
        return aggregated_parameters

    def aggregate_evaluate(self, server_round, results, failures):
        start_time = time.time()
        loss_aggregated = super().aggregate_evaluate(server_round, results, failures)[0]
        comm_time = time.time() - start_time
        self.communication_times.append({'round': server_round, 'time': comm_time, 'total_round_time': time.time() - self.round_start_time})
        metrics = [res.metrics for _, res in results]
        self.global_metrics.append({
            'round': server_round,
            'rmse': loss_aggregated,
            'mae': np.mean([m['mae'] for m in metrics]),
            'r2': np.mean([m['r2'] for m in metrics])
        })
        return loss_aggregated, {}

    def finalize(self):
        for metric in ['rmse', 'mae', 'r2', 'training_time']:
            plt.figure(figsize=(8, 6))
            for cid in self.client_metrics[0]['clients'].keys():
                values = [m['clients'][cid][metric] for m in self.client_metrics]
                plt.plot(range(1, N_ROUNDS + 1), values, label=f'{cid} {metric.capitalize()}', marker='o')
            plt.xlabel('Federated Round')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} Across Clients - Linear Regression')
            plt.legend()
            plt.savefig(os.path.join(OUTPUT_DIR, f'linear_regression_{metric}_clients.png'))
            plt.close()

        plt.figure(figsize=(8, 6))
        rounds = [v['round'] for v in self.weight_variances]
        variances = [v['variance'] for v in self.weight_variances]
        plt.plot(rounds, variances, marker='o')
        plt.xlabel('Federated Round')
        plt.ylabel('Weight Variance')
        plt.title('Weight Variance - Linear Regression')
        plt.savefig(os.path.join(OUTPUT_DIR, 'linear_regression_weight_variance.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        rounds = [t['round'] for t in self.communication_times]
        comm_times = [t['time'] for t in self.communication_times]
        agg_times = [t['time'] for t in self.aggregation_times]
        total_round_times = [t['total_round_time'] for t in self.communication_times]
        plt.plot(rounds, comm_times, label='Communication Time', marker='o')
        plt.plot(rounds, agg_times, label='Aggregation Time', marker='o')
        plt.plot(rounds, total_round_times, label='Total Round Time', marker='o')
        plt.xlabel('Federated Round')
        plt.ylabel('Time (seconds)')
        plt.title('Federated System Times - Linear Regression')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'linear_regression_system_times.png'))
        plt.close()

        report = f"# Federated Learning Report - Linear Regression\n\n"
        report += f"**Federated Rounds**: {N_ROUNDS}\n\n"
        report += "## Global Performance\n"
        for m in self.global_metrics:
            report += f"- **Round {m['round']}**:\n"
            report += f"  - RMSE: {m['rmse']:.4f}\n"
            report += f"  - MAE: {m['mae']:.4f}\n"
            report += f"  - R²: {m['r2']:.4f}\n"
        
        report += "\n## Client Performance\n"
        for cid in self.client_metrics[0]['clients'].keys():
            report += f"### {cid}\n"
            for m in self.client_metrics:
                report += f"- **Round {m['round']}**:\n"
                report += f"  - RMSE: {m['clients'][cid]['rmse']:.4f}\n"
                report += f"  - MAE: {m['clients'][cid]['mae']:.4f}\n"
                report += f"  - R²: {m['clients'][cid]['r2']:.4f}\n"
                report += f"  - Training Time: {m['clients'][cid]['training_time']:.2f} seconds\n"
        
        report += "\n## System Times\n"
        total_system_time = sum(t['total_round_time'] for t in self.communication_times)
        report += f"- **Total System Time**: {total_system_time:.2f} seconds\n"
        for r in range(1, N_ROUNDS + 1):
            comm_time = next(t['time'] for t in self.communication_times if t['round'] == r)
            agg_time = next(t['time'] for t in self.aggregation_times if t['round'] == r)
            total_round_time = next(t['total_round_time'] for t in self.communication_times if t['round'] == r)
            report += f"- **Round {r}**:\n"
            report += f"  - Communication Time: {comm_time:.2f} seconds\n"
            report += f"  - Aggregation Time: {agg_time:.2f} seconds\n"
            report += f"  - Total Round Time: {total_round_time:.2f} seconds\n"
        
        report += "\n## Analysis\n"
        report += "- **Convergence**: Global RMSE stabilizes quickly due to the simplicity of Linear Regression.\n"
        report += "- **Stability**: Low weight variance indicates consistent updates across clients.\n"
        report += "- **Environment Impact**: Raspberry Pi 3B may have slightly higher training times due to ARM hardware.\n"
        report += "- **Communication**: Network latency impacts communication times, especially for Raspberry Pi.\n"
        report += f"- **Plots**: See `linear_regression_rmse_clients.png`, `mae_clients.png`, `r2_clients.png`, `training_time_clients.png`, `weight_variance.png`, `system_times.png`.\n"
        
        with open(os.path.join(OUTPUT_DIR, 'fl_report_linear_regression.md'), 'w') as f:
            f.write(report)

# Iniciar servidor
total_start_time = time.time()
strategy = LinearRegressionStrategy()
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
    strategy=strategy
)
strategy.finalize()
total_time = time.time() - total_start_time
with open(os.path.join(OUTPUT_DIR, 'total_server_time.txt'), 'w') as f:
    f.write(f"Total Server Execution Time: {total_time:.2f} seconds")