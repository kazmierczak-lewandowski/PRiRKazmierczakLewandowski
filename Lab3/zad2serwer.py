import pickle
import socket
import time

import numpy as np
from PIL import Image

image_path = "401.jpg"
def receive_all(sock):
    length = int.from_bytes(sock.recv(4), byteorder='big') 
    data = b''
    while len(data) < length:
        packet = sock.recv(4096)
        if not packet:
            break
        data += packet
    return pickle.loads(data)

def send_all(sock, data):
    data = pickle.dumps(data)
    sock.sendall(len(data).to_bytes(4, byteorder='big')) 
    sock.sendall(data)


def split_image(image, n_clients):
    if n_clients <= 0:
        raise ValueError("Liczba klientów musi być większa od zera")
    if n_clients > image.height:
        raise ValueError("Liczba klientów nie może być większa od wysokości obrazu")

    image_array = np.array(image)
    return np.array_split(image_array, n_clients, axis=0)


def merge_image(fragments):
    if not fragments:
        raise ValueError("Lista fragmentów nie może być pusta")

    width = fragments[0].shape[1]
    if any(fragment.shape[1] != width for fragment in fragments):
        raise ValueError(
            "Wszystkie fragmenty muszą mieć tę samą szerokość"
        )

    return Image.fromarray(np.concatenate(fragments, axis=0))

def server_main(image_path, n_clients):
    process_start_time = time.perf_counter()
    image = Image.open(image_path)
    fragments = split_image(image, n_clients)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('172.20.10.2', 2040))
    server_socket.listen(n_clients)
    print("Serwer nasłuchuje...")
    processed_fragments = []
    for i in range(n_clients):
        client_socket, client_adress = server_socket.accept()
        print(f"Połączono z klientem {i+1}: {client_adress}")
        start_time = time.perf_counter()
        send_all(client_socket, fragments[i])
        processed_fragment = receive_all(client_socket)
        elapsed_time = time.perf_counter() - start_time
        print(
            f"Czas wysłania fragmentu i otrzymania wyniku od klienta "
            f"{i+1}: {elapsed_time:.4f} s"
        )
        processed_fragments.append(processed_fragment)
        client_socket.close()
    result_image = merge_image(processed_fragments)
    result_image.save("processed_image.png")
    print("Obraz przetworzony zapisany jako processed_image.png")
    process_time = time.perf_counter() - process_start_time
    print(f"Czas całego procesu: {process_time:.4f} s")


if __name__ == "__main__":
    server_main(image_path, 1)