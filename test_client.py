"""
Test client for the PowerGrid Electric Voice API.

Captures mic audio, sends it to the WebSocket server, and plays back
the assistant's audio responses.

Usage:
    python test_client.py                          # localhost
    python test_client.py wss://xyz.ngrok-free.app/ws  # ngrok
"""

import asyncio
import base64
import json
import sys
import threading
import queue

import pyaudio
import websockets

SAMPLE_RATE = 24_000
CHANNELS = 1
FRAME_DURATION_MS = 100
FRAMES_PER_BUFFER = SAMPLE_RATE * FRAME_DURATION_MS // 1000

playback_queue: queue.Queue[bytes | None] = queue.Queue()


def playback_worker():
    """Plays received audio chunks through speakers."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE,
        output=True, frames_per_buffer=FRAMES_PER_BUFFER,
    )
    try:
        while True:
            chunk = playback_queue.get()
            if chunk is None:
                break
            stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


async def main(url: str):
    print(f"Connecting to {url} ...")

    # Start playback thread
    player = threading.Thread(target=playback_worker, daemon=True)
    player.start()

    async with websockets.connect(url) as ws:
        print("Connected! Speak into your mic. Ctrl+C to quit.\n")

        async def send_audio():
            """Capture mic and send to server."""
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=FRAMES_PER_BUFFER,
            )
            loop = asyncio.get_event_loop()
            try:
                while True:
                    pcm = await loop.run_in_executor(
                        None, stream.read, FRAMES_PER_BUFFER, False,
                    )
                    b64 = base64.b64encode(pcm).decode()
                    await ws.send(json.dumps({"type": "audio", "audio": b64}))
            except Exception:
                pass
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()

        async def receive():
            """Receive messages from server."""
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("type", "")

                    if mtype == "audio":
                        pcm = base64.b64decode(msg["audio"])
                        playback_queue.put(pcm)

                    elif mtype == "transcript":
                        role = msg.get("role", "?")
                        text = msg.get("text", "")
                        label = "[you]" if role == "user" else "[assistant]"
                        print(f"{label} {text}")

                    elif mtype == "error":
                        print(f"[error] {msg.get('message', '')}")
            except Exception as e:
                print(f"[disconnected] {e}")

        done, pending = await asyncio.wait(
            [asyncio.create_task(send_audio()), asyncio.create_task(receive())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

    playback_queue.put(None)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws"
    try:
        asyncio.run(main(url))
    except KeyboardInterrupt:
        print("\nBye!")
        playback_queue.put(None)
