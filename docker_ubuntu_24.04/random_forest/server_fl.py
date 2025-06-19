import socket
import threading
import json
import time

HOST = '0.0.0.0'
PORT_RECEIVE = 5000  # Porta para receber dos clientes
PORT_SEND = 5001     # Porta para enviar para os clientes

clients_params = {}
clients_addresses = {}
lock = threading.Lock()

def fed_avg(params_list):
    aggregated = {}
    keys = params_list[0].keys()

    for key in keys:
        values = [p[key] for p in params_list if p[key] is not None]

        if all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = round(sum(values) / len(values))
        else:
            aggregated[key] = values[0]  # Keep first for categorical

    return aggregated


def listener_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT_RECEIVE))
    server.listen()

    print(f"[LISTENER] Aguardando conexões na porta {PORT_RECEIVE}...")

    while True:
        conn, addr = server.accept()
        client_id = addr[0]
        print(f"[LISTENER] Conectado a {addr}")

        data = conn.recv(8192).decode()
        params = json.loads(data)

        with lock:
            clients_params[client_id] = params
            clients_addresses[client_id] = addr[0]

        print(f"[LISTENER] Parâmetros recebidos de {client_id}: {params}")

        conn.close()


def federated_loop():
    while True:
        print("[FEDAVG] Aguardando próximo ciclo de federação (20 minutos)...\n")
        time.sleep(1200)  # 20 minutos

        with lock:
            if len(clients_params) >= 2:
                print("[FEDAVG] Executando FedAvg...")
                params_list = list(clients_params.values())
                aggregated_params = fed_avg(params_list)
                print(f"[FEDAVG] Parâmetros agregados: {aggregated_params}")

                for client_ip in clients_addresses.values():
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.connect((client_ip, PORT_SEND))
                            s.sendall(json.dumps(aggregated_params).encode())
                        print(f"[FEDAVG] Enviado para {client_ip}")
                    except Exception as e:
                        print(f"[FEDAVG] Erro ao enviar para {client_ip}: {e}")
            else:
                print("[FEDAVG] Clientes insuficientes para agregação.")

if __name__ == "__main__":
    t1 = threading.Thread(target=listener_thread)
    t2 = threading.Thread(target=federated_loop)

    t1.start()
    t2.start()

    t1.join()
    t2.join()