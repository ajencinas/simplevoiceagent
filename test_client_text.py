"""
Text-based test client for the PowerGrid Electric Voice API.

Type messages that are sent directly as text. The assistant's audio
responses are played through your speakers. Shows latency measurements.

Usage:
    python test_client_text.py                              # localhost
    python test_client_text.py wss://xyz.ngrok-free.app/ws  # ngrok
"""

import asyncio
import base64
import ctypes
import ctypes.util
import json
import os
import sys
import threading
import time
import queue

import pyaudio
import websockets

# Suppress ALSA/JACK warnings
_ALSA_ERR_TYPE = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
_alsa_err_cb = _ALSA_ERR_TYPE(lambda *_: None)
try:
    _asound = ctypes.cdll.LoadLibrary(ctypes.util.find_library("asound") or "libasound.so.2")
    _asound.snd_lib_error_set_handler(_alsa_err_cb)
except Exception:
    pass

SAMPLE_RATE = 24_000
CHANNELS = 1
FRAMES_PER_BUFFER = SAMPLE_RATE * 100 // 1000

playback_queue: queue.Queue[bytes | None] = queue.Queue()

# Timing
_send_time: float = 0.0
_first_audio_received = False


def _open_pyaudio() -> pyaudio.PyAudio:
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    os.dup2(devnull, 2)
    try:
        pa = pyaudio.PyAudio()
    finally:
        os.dup2(old, 2)
        os.close(old)
        os.close(devnull)
    return pa


def playback_worker():
    pa = _open_pyaudio()
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
    print(f"[client] Connecting to {url} ...")

    player = threading.Thread(target=playback_worker, daemon=True)
    player.start()

    async with websockets.connect(url) as ws:
        print("[client] Connected! Type your messages below. Ctrl+C to quit.\n")

        async def receive():
            global _first_audio_received
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("type", "")

                    if mtype == "audio":
                        if not _first_audio_received and _send_time > 0:
                            ttfb = (time.perf_counter() - _send_time) * 1000
                            print(f"[client] First audio byte: {ttfb:.0f}ms")
                            _first_audio_received = True
                        pcm = base64.b64decode(msg["audio"])
                        playback_queue.put(pcm)

                    elif mtype == "transcript":
                        role = msg.get("role", "?")
                        text = msg.get("text", "")
                        label = "[you]" if role == "user" else "[assistant]"
                        if role == "assistant" and _send_time > 0:
                            total = (time.perf_counter() - _send_time) * 1000
                            print(f"[client] Full response: {total:.0f}ms")
                        print(f"{label} {text}")

                    elif mtype == "error":
                        print(f"[client] ERROR: {msg.get('message', '')}")
            except Exception as e:
                print(f"[client] Disconnected: {e}")

        async def send_text():
            global _send_time, _first_audio_received
            loop = asyncio.get_event_loop()
            try:
                while True:
                    text = await loop.run_in_executor(None, lambda: input("\n> "))
                    if not text.strip():
                        continue

                    _first_audio_received = False
                    _send_time = time.perf_counter()
                    await ws.send(json.dumps({"type": "text", "text": text}))
                    print("[client] Sent, waiting for response...")

            except (EOFError, KeyboardInterrupt):
                pass

        done, pending = await asyncio.wait(
            [asyncio.create_task(send_text()), asyncio.create_task(receive())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

    playback_queue.put(None)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws"
    # Auto-fix common URL mistakes
    url = url.replace("https://", "wss://").replace("http://", "ws://")
    if not url.endswith("/ws"):
        url = url.rstrip("/") + "/ws"
    try:
        asyncio.run(main(url))
    except KeyboardInterrupt:
        print("\nBye!")
        playback_queue.put(None)
