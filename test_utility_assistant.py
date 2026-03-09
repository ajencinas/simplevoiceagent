"""
Test harness for the Electric Utility Voice Assistant
=====================================================
Generates spoken audio via OpenAI TTS, then feeds it into the Realtime API
over WebSocket — no microphone needed.

Usage:
    export OPENAI_API_KEY="sk-..."
    python test_utility_assistant.py
"""

import os
import sys
import json
import base64
import threading
import time

import pyaudio
import websocket
from openai import OpenAI

# ── Re-use all the tools / DB / handlers from the main app ──────────────────

from utility_voice_assistant import (
    OPENAI_API_KEY,
    WS_URL,
    WS_HEADERS,
    SAMPLE_RATE,
    CHANNELS,
    FRAMES_PER_BUFFER,
    SYSTEM_INSTRUCTIONS,
    TOOLS,
    TOOL_DISPATCH,
    session_state,
)

# ── Test scenario: sequence of things the "caller" says ──────────────────────

TEST_UTTERANCES = [
    "Hi, I'd like to check my electric bill please.",
    "My account number is 1 2 3 4 5.",
    "My pin is 9 9 9 9.",
    "Can you tell me my current balance and when it's due?",
    "I'd like to pay fifty dollars with my credit card.",
    "Yes, please go ahead.",
    "That's all, thank you!",
]

# ── Generate test audio using OpenAI TTS ─────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)


def text_to_pcm16_24k(text: str) -> bytes:
    """Use OpenAI TTS to generate PCM16 24kHz mono audio from text."""
    print(f'  [tts] Generating: "{text}"')
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions="Speak naturally as a phone caller. Be conversational.",
        response_format="pcm",
    )
    return response.content


def generate_all_test_audio():
    """Pre-generate all test utterances as PCM16 audio."""
    print("[setup] Generating test audio with OpenAI TTS...")
    audio_clips = []
    for utterance in TEST_UTTERANCES:
        pcm = text_to_pcm16_24k(utterance)
        audio_clips.append(pcm)
        print(f"  [tts] Got {len(pcm)} bytes ({len(pcm) / SAMPLE_RATE / 2:.1f}s)")
    print(f"[setup] Generated {len(audio_clips)} audio clips.\n")
    return audio_clips


# ── Playback (separate from the main app to avoid import side-effects) ──────

playback_queue: "queue.Queue[bytes | None]" = None  # created at runtime
import queue
playback_idle = threading.Event()

def playback_worker():
    """Background thread that plays response audio through speakers."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )
    try:
        while True:
            chunk = playback_queue.get()
            if chunk is None:
                break
            playback_idle.clear()
            stream.write(chunk)
            if playback_queue.empty():
                playback_idle.set()
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def wait_for_playback_idle(timeout_s: float = 30.0, settle_s: float = 0.25) -> bool:
    """Wait until playback queue drains and current stream write has finished."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if playback_queue.empty() and playback_idle.is_set():
            time.sleep(settle_s)
            if playback_queue.empty() and playback_idle.is_set():
                return True
        time.sleep(0.05)
    return False


# ── WebSocket test client ────────────────────────────────────────────────────

class TestRunner:
    def __init__(self, audio_clips: list[bytes], play_user_audio: bool = True):
        self.audio_clips = audio_clips
        self.play_user_audio = play_user_audio
        self.clip_index = 0
        self.ws = None
        self.response_complete = threading.Event()
        self.session_ready = threading.Event()
        self.waiting_for_tool_response = False

    def send_session_update(self):
        """Configure session — must include type: realtime."""
        event = {
            "type": "session.update",
            "session": {
                "instructions": SYSTEM_INSTRUCTIONS,
                "tools": TOOLS,
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gpt-4o-mini-transcribe",
                },
                "turn_detection": None,  # disabled — we control turn-taking
                "output_audio_format": "pcm16",
                "voice": "coral",
            },
        }
        self.ws.send(json.dumps(event))

    def send_audio_clip(self, pcm_data: bytes):
        """Send a full PCM audio clip in chunks, commit, then request response."""
        chunk_size = FRAMES_PER_BUFFER * 2  # 2 bytes per int16 sample
        total_sent = 0
        chunk_duration_s = FRAMES_PER_BUFFER / SAMPLE_RATE
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i : i + chunk_size]
            if self.play_user_audio:
                playback_queue.put(chunk)
            b64 = base64.b64encode(chunk).decode("utf-8")
            self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64,
            }))
            total_sent += len(chunk)
            # Stream in real-time to keep user playback and server ingest aligned.
            time.sleep(chunk_duration_s)

        print(f"[send]   Sent {total_sent} bytes of audio")
        # Small delay to let the server ingest the buffer
        time.sleep(0.2)
        self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("[send]   Buffer committed")
        time.sleep(0.1)
        self.ws.send(json.dumps({"type": "response.create"}))
        print("[send]   Response requested")

    def on_open(self, ws):
        print("[ws] Connected to OpenAI Realtime API")
        self.send_session_update()

    def on_message(self, ws, message):
        event = json.loads(message)
        etype = event.get("type", "")

        if etype == "session.created":
            sid = event.get("session", {}).get("id", "N/A")
            print(f"[session] Created: {sid}")

        elif etype == "session.updated":
            print("[session] Updated — ready.\n")
            self.session_ready.set()
            # Now that session is configured, request greeting
            self.ws.send(json.dumps({"type": "response.create"}))

        elif etype == "response.audio.delta":
            audio_b64 = event.get("delta", "")
            if audio_b64:
                pcm = base64.b64decode(audio_b64)
                playback_queue.put(pcm)

        elif etype == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "").strip()
            if transcript:
                print(f"[you]    {transcript}")

        elif etype == "response.audio_transcript.delta":
            text = event.get("delta", "")
            if text:
                print(text, end="", flush=True)

        elif etype == "response.audio_transcript.done":
            print()

        elif etype == "response.function_call_arguments.done":
            call_id = event.get("call_id", "")
            name = event.get("name", "")
            arguments = event.get("arguments", "{}")
            print(f"\n[tool]   {name}({arguments})")

            func = TOOL_DISPATCH.get(name)
            if func:
                try:
                    args = json.loads(arguments)
                    result = func(**args)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown function: {name}"}
            print(f"[tool]   -> {json.dumps(result)}")

            self.waiting_for_tool_response = True

            ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }))
            # Let server process the item before requesting response
            time.sleep(0.1)
            ws.send(json.dumps({"type": "response.create"}))

        elif etype == "response.done":
            response = event.get("response", {})
            output = response.get("output", [])
            has_audio = any(item.get("type") == "message" for item in output)
            if has_audio:
                self.waiting_for_tool_response = False
                self.response_complete.set()

        elif etype == "error":
            err = event.get("error", {})
            print(f"\n[error]  {err.get('type', 'unknown')}: {err.get('message', '')}")
            self.response_complete.set()

        # Silently ignore known noisy events
        elif etype in (
            "response.created",
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "conversation.item.created",
            "input_audio_buffer.committed",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "conversation.item.input_audio_transcription.delta",
            "rate_limits.updated",
        ):
            pass
        else:
            print(f"[event]  {etype}")

    def on_error(self, ws, error):
        print(f"\n[ws] Error: {error}")

    def on_close(self, ws, code, msg):
        print(f"\n[ws] Disconnected (code={code})")
        playback_queue.put(None)

    def run(self):
        global playback_queue
        playback_queue = queue.Queue()
        playback_idle.set()

        self.ws = websocket.WebSocketApp(
            WS_URL,
            header=WS_HEADERS,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        # Start audio playback thread
        threading.Thread(target=playback_worker, daemon=True).start()

        # Start websocket in background thread
        ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        ws_thread.start()

        # Wait for session to be configured
        if not self.session_ready.wait(timeout=15):
            print("[test] Timeout waiting for session setup!")
            return

        # Wait for the initial greeting to finish
        print("[test] Waiting for greeting...")
        if not self.response_complete.wait(timeout=30):
            print("[test] Timeout waiting for greeting!")
            return
        wait_for_playback_idle(timeout_s=30)

        # Send each test utterance one at a time
        for idx, clip in enumerate(self.audio_clips):
            self.response_complete.clear()

            print(f"\n{'='*60}")
            print(f"[test] Step {idx + 1}/{len(self.audio_clips)}: \"{TEST_UTTERANCES[idx]}\"")
            print(f"{'='*60}")

            self.send_audio_clip(clip)

            # Wait for full response (audio reply, not just tool call)
            if not self.response_complete.wait(timeout=60):
                print("[test] Timeout waiting for response, stopping.")
                break

            wait_for_playback_idle(timeout_s=45)

        print("\n" + "=" * 60)
        print("[test] Test complete!")
        print("=" * 60)
        self.ws.close()
        time.sleep(1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not OPENAI_API_KEY:
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    print("=" * 60)
    print("  PowerGrid Electric — Voice Assistant TEST HARNESS")
    print("  Sends pre-generated TTS audio instead of live mic")
    print("=" * 60)
    print()

    # Generate all test audio clips via OpenAI TTS
    audio_clips = generate_all_test_audio()

    # Reset session state
    session_state["authenticated"] = False
    session_state["account"] = None

    # Run the test
    runner = TestRunner(audio_clips)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[test] Interrupted.")
        if playback_queue:
            playback_queue.put(None)


if __name__ == "__main__":
    main()
