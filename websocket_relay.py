import asyncio
import websockets
import os

clients = set()

async def handler(websocket, path):
    clients.add(websocket)
    try:
        async for message in websocket:
            # Relay the message to all other clients
            for client in clients:
                if client != websocket:
                    await client.send(message)
    finally:
        clients.remove(websocket)

PORT = int(os.environ.get("PORT", 10000))

async def main():
    print(f"WebSocket relay server listening on port {PORT}")
    async with websockets.serve(handler, "0.0.0.0", PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main()) 