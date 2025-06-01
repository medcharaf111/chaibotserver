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
start_server = websockets.serve(handler, "0.0.0.0", PORT)

print(f"WebSocket relay server listening on port {PORT}")

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever() 