import socket, threading

clients = []

def handle_client(conn, addr):
    print(f"Connected: {addr}")
    clients.append(conn)
    try:
        while True:
            data = conn.recv(1024)
            if not data: break
            # Relay to all other clients
            for c in clients:
                if c != conn:
                    c.sendall(data)
    finally:
        clients.remove(conn)
        conn.close()

s = socket.socket()
s.bind(('0.0.0.0', 10000))  # Use port 10000 or any open port
s.listen()
print("Server listening on port 10000")
while True:
    conn, addr = s.accept()
    threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start() 