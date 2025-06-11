# Intel Lab Federated Learning Repository

This repository implements **Federated Learning (FL)** with three machine learning models (**Linear Regression**, **Random Forest**, and **CNN**) on the Intel Lab Sensor Data dataset to predict temperature. The setup supports three environments: **Local (Ubuntu 22.04)**, **Docker (Ubuntu 24.04)**, and **Raspberry Pi 3B (Ubuntu)**, using the Flower framework for FL.

## Repository Structure

```
intel-lab-fl-repo/
├── local_ubuntu_22.04/          # Scripts and setup for Ubuntu 22.04
├── docker_ubuntu_24.04/         # Docker setups for Ubuntu 24.04
├── raspberry_pi_ubuntu/         # Scripts and setup for Raspberry Pi
├── .gitignore                   # Git ignore file
├── README.md                    # This file
```

Each environment directory contains:
- `scripts/` or model-specific folders: Client and server scripts for FL.
- `download_dataset.py`: Script to download the dataset from Google Drive.
- `setup.sh`: Installation script for dependencies.
- `requirements.txt`: Python dependencies (for Local and Raspberry Pi).
- `data/`: Placeholder for `intel_lab_data_cleaned.csv` (not included in Git).
- `ml_results_<model>/`: Output directories for results.

## Prerequisites

- **Git**: Installed on all environments (`sudo apt install git`).
- **Python 3.10+**: Required for all environments.
- **Docker**: Required for `docker_ubuntu_24.04` (Ubuntu 24.04).
- **Network**: Server runs on `192.168.1.100:8080` (adjust as needed).
- **Google Drive Access**: Ensure the dataset file is accessible via a shareable link.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/intel-lab-fl-repo.git
cd intel-lab-fl-repo
```

### 2. Download the Dataset

For each environment, run the `download_dataset.py` script to download `intel_lab_data_cleaned.csv`. Replace `GOOGLE_DRIVE_ID` in `download_dataset.py` with the actual file ID from Google Drive.

### 3. Local (Ubuntu 22.04)

1. Navigate to the environment:
   ```bash
   cd local_ubuntu_22.04
   ```

2. Install dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Download the dataset:
   ```bash
   python3 download_dataset.py
   ```

4. Start the server (e.g., Linear Regression):
   ```bash
   python3 scripts/fl_server_linear_regression.py
   ```

5. In a new terminal, start the client:
   ```bash
   python3 scripts/fl_client_linear_regression.py client1 192.168.1.100:8080
   ```

6. Repeat for Random Forest and CNN.

### 4. Docker (Ubuntu 24.04)

1. Install Docker:
   ```bash
   sudo apt update
   sudo apt install docker.io
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

2. Navigate to the model folder (e.g., Linear Regression):
   ```bash
   cd docker_ubuntu_24.04/linear_regression
   ```

3. Download the dataset:
   ```bash
   python3 ../download_dataset.py
   ```

4. Build and run the client:
   ```bash
   chmod +x ../setup.sh
   ../setup.sh linear_regression
   ```

5. Ensure the server is running in `local_ubuntu_22.04` before starting the client.

6. Repeat for Random Forest and CNN.

### 5. Raspberry Pi 3B (Ubuntu)

1. Navigate to the environment:
   ```bash
   cd raspberry_pi_ubuntu
   ```

2. Install dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Download the dataset:
   ```bash
   python3 download_dataset.py
   ```

4. Start the client (e.g., CNN):
   ```bash
   python3 scripts/fl_client_cnn.py client3 192.168.1.100:8080
   ```

5. Ensure the server is running in `local_ubuntu_22.04`.

6. Repeat for Linear Regression and Random Forest. **Note**: For CNN, reduce `N_EPOCHS` (e.g., to 50) in `fl_client_cnn.py` if performance is slow.

## Network Configuration

1. **Identify Server IP**:
   - On Ubuntu 22.04, run:
     ```bash
     ip addr show
     ```
   - Update `SERVER_ADDRESS` in client scripts if different from `192.168.1.100:8080`.

2. **Firewall**:
   - Allow port 8080:
     ```bash
     sudo ufw allow 8080
     sudo ufw status
     ```

3. **Test Connectivity**:
   - From clients:
     ```bash
     ping 192.168.1.100
     nc -zv 192.168.1.100 8080
     ```

## Output

Each model generates results in `ml_results_<model>/`:
- **Client**: Metrics, times (`total_time_<client_id>.txt`), plots (scatter, feature importance for Random Forest).
- **Server**: Reports (`fl_report_<model>.md`), times (`total_server_time.txt`), plots (metrics, weight variance, system times).

## Notes

- **Dataset**: Downloaded via `download_dataset.py`. Ensure `GOOGLE_DRIVE_ID` is correct.
- **Raspberry Pi**: May require longer training times, especially for CNN. Ensure PyTorch and Flower are installed for ARM.
- **Adjustments**: Modify `N_EPOCHS` (CNN), `N_ESTIMATORS` (Random Forest), or `N_ROUNDS` in scripts for performance tuning.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/<name>`).
3. Commit changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature/<name>`).
5. Create a Pull Request.

## License

MIT License (see `LICENSE` file if added).