import pickle
import socket

from zad1 import apply_sobel


def client_main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('172.20.10.2', 2040))
    fragment = receive_all(client_socket)
    processed_fragment = apply_sobel(fragment)
    send_all(client_socket, processed_fragment)
    client_socket.close()
    print("Fragment przetworzony i wysłany z powrotem do serwera")


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

client_main()