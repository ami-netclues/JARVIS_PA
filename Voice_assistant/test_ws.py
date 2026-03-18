import asyncio
import websockets
import json

async def test_enroll():
    uri = "ws://127.0.0.1:8000/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            # Initial status
            await websocket.recv()
            
            # Send enroll action
            action = {
                "action": "enroll_step",
                "step": 1,
                "target": "I use Jarvis every single day",
                "slot": 1,
                "name": "TestUser"
            }
            await websocket.send(json.dumps(action))
            print("Sent enroll_step action")
            
            # Use a timeout to wait for enroll_status
            try:
                while True:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(msg)
                    print(f"Received from master: {data}")
                    if data.get("type") == "enroll_status" and data.get("status") == "recording":
                        print("YAY! Backend started recording.")
                        break
            except asyncio.TimeoutError:
                print("Timed out waiting for response")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_enroll())
