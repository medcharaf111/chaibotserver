import asyncio
import websockets
import os

clients = set()

async def handler(websocket, path):
    print(f"New connection: {websocket.remote_address}")
    clients.add(websocket)
    print(f"Total clients: {len(clients)}")
    try:
        async for message in websocket:
            print(f"Received: {message}")
            for client in clients:
                if client != websocket:
                    print(f"Relaying to: {client.remote_address}")
                    await client.send(message)
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        clients.remove(websocket)
        print(f"Connection closed: {websocket.remote_address}")
        print(f"Total clients: {len(clients)}")

PORT = int(os.environ.get("PORT", 10000))

async def main():
    print(f"WebSocket relay server listening on port {PORT}")
    async with websockets.serve(handler, "0.0.0.0", PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main()) 