import socket
import json
import os
import threading
import time

SERVER_IP = '192.168.0.17'
PORT_RECEIVE = 5001
PORT_SEND = 5000

CLIENT_ID = 'client2'
params_file = f'../ml_results_random_forest/params_{CLIENT_ID}.json'


def load_params():
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            return json.load(f)
    else:
        print("[CLIENT] Nenhum parâmetro encontrado localmente.")
        return None


def save_params(params):
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    print("[CLIENT] Parâmetros atualizados e salvos.")


def send_params_loop():
    while True:
        params = load_params()
        if params is None:
            print("[CLIENT] Aguardando parâmetros locais...")
            time.sleep(60)
            continue

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((SERVER_IP, PORT_SEND))
                s.sendall(json.dumps(params).encode())
            print("[CLIENT] Parâmetros enviados ao servidor.")
        except Exception as e:
            print(f"[CLIENT] Erro ao enviar para servidor: {e}")

        time.sleep(600)  # Enviar a cada 10 minutos


def receive_params_thread():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('0.0.0.0', PORT_RECEIVE))
    listener.listen()

    print(f"[CLIENT] Aguardando parâmetros federados na porta {PORT_RECEIVE}...")

    while True:
        conn, addr = listener.accept()
        data = conn.recv(8192).decode()
        aggregated_params = json.loads(data)
        print(f"[CLIENT] Parâmetros federados recebidos: {aggregated_params}")
        save_params(aggregated_params)

        conn.close()

if __name__ == "__main__":
    t1 = threading.Thread(target=send_params_loop)
    t2 = threading.Thread(target=receive_params_thread)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
