"""
Electric Utility Voice AI Assistant — POC
==========================================
Uses OpenAI Realtime API (gpt-realtime) over WebSocket for speech-to-speech
interaction.  Simulates authentication and bill-pay for a fictitious
electric utility ("PowerGrid Electric").

Echo handling (--echo flag):
    block — Mute mic while assistant speaks.  No barge-in but no echo.
            Use with laptop speakers.  (default)
    off   — Mic always open.  Use when echo cancellation is handled
            elsewhere (headphones, telephony, PulseAudio module-echo-cancel,
            WebRTC, etc.)

Requirements:
    pip install websocket-client pyaudio

Usage:
    export OPENAI_API_KEY="sk-..."
    python utility_voice_assistant.py                # block (default)
    python utility_voice_assistant.py --echo off     # mic always open
"""

import argparse
import ctypes
import ctypes.util
import os
import sys
import json
import base64
import threading
import queue
import time
from datetime import date

import pyaudio
import websocket

# Suppress ALSA warnings.  Callback must stay alive at module level.
_ALSA_ERR_TYPE = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
_alsa_err_cb = _ALSA_ERR_TYPE(lambda *_: None)
try:
    _asound = ctypes.cdll.LoadLibrary(ctypes.util.find_library("asound") or "libasound.so.2")
    _asound.snd_lib_error_set_handler(_alsa_err_cb)
except Exception:
    pass


def _open_pyaudio() -> pyaudio.PyAudio:
    """Create PyAudio instance with JACK/ALSA stderr noise suppressed."""
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


# ── Configuration ────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-realtime"
WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
WS_HEADERS = [
    f"Authorization: Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta: realtime=v1",
]
SAMPLE_RATE = 24_000
CHANNELS = 1
FRAME_DURATION_MS = 100
FRAMES_PER_BUFFER = SAMPLE_RATE * FRAME_DURATION_MS // 1000

ECHO_MODE = "block"  # "block" | "off"

# ── Fake customer database (POC) ────────────────────────────────────────────

CUSTOMERS = {
    "12345": {
        "name": "John Smith",
        "pin": "9999",
        "address": "742 Evergreen Terrace",
        "account_number": "12345",
        "balance_due": 147.83,
        "due_date": "2026-03-15",
        "last_payment": 125.00,
        "last_payment_date": "2026-01-20",
        "plan": "Residential Standard",
        "kwh_used": 1082,
    },
    "67890": {
        "name": "Jane Smith",
        "pin": "1234",
        "address": "123 Main St",
        "account_number": "67890",
        "balance_due": 89.21,
        "due_date": "2026-03-10",
        "last_payment": 92.50,
        "last_payment_date": "2026-01-18",
        "plan": "Residential Economy",
        "kwh_used": 743,
    },
}

session_state = {"authenticated": False, "account": None}

# ── Tool implementations ────────────────────────────────────────────────────

def lookup_account(account_number: str) -> dict:
    acct = CUSTOMERS.get(account_number)
    if acct:
        return {"found": True, "name": acct["name"], "account_number": account_number}
    return {"found": False, "message": f"No account found for number {account_number}."}


def authenticate(account_number: str, pin: str) -> dict:
    acct = CUSTOMERS.get(account_number)
    if not acct:
        return {"success": False, "message": "Account not found."}
    if acct["pin"] != pin:
        return {"success": False, "message": "Incorrect PIN. Please try again."}
    session_state["authenticated"] = True
    session_state["account"] = account_number
    return {"success": True, "message": f"Welcome, {acct['name']}! You are now authenticated."}


def get_bill_summary(account_number: str) -> dict:
    if not session_state["authenticated"] or session_state["account"] != account_number:
        return {"error": "Please authenticate first."}
    acct = CUSTOMERS.get(account_number)
    if not acct:
        return {"error": "Account not found."}
    return {
        "name": acct["name"],
        "balance_due": acct["balance_due"],
        "due_date": acct["due_date"],
        "kwh_used": acct["kwh_used"],
        "plan": acct["plan"],
        "last_payment": acct["last_payment"],
        "last_payment_date": acct["last_payment_date"],
    }


def make_payment(account_number: str, amount: float, payment_method: str) -> dict:
    if not session_state["authenticated"] or session_state["account"] != account_number:
        return {"error": "Please authenticate first."}
    acct = CUSTOMERS.get(account_number)
    if not acct:
        return {"error": "Account not found."}
    if amount <= 0:
        return {"error": "Payment amount must be positive."}
    if amount > acct["balance_due"]:
        return {"error": f"Amount ${amount:.2f} exceeds balance due of ${acct['balance_due']:.2f}."}
    acct["balance_due"] = round(acct["balance_due"] - amount, 2)
    acct["last_payment"] = amount
    acct["last_payment_date"] = date.today().isoformat()
    return {
        "success": True,
        "message": f"Payment of ${amount:.2f} via {payment_method} processed successfully.",
        "confirmation_number": "PG-2026-" + account_number + "-" + str(int(time.time()) % 100000),
        "remaining_balance": acct["balance_due"],
    }


TOOL_DISPATCH = {
    "lookup_account": lookup_account,
    "authenticate": authenticate,
    "get_bill_summary": get_bill_summary,
    "make_payment": make_payment,
}

# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "name": "lookup_account",
        "description": "Look up a customer account by account number to verify it exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_number": {"type": "string", "description": "The customer's account number."}
            },
            "required": ["account_number"],
        },
    },
    {
        "type": "function",
        "name": "authenticate",
        "description": "Authenticate a customer using their account number and 4-digit PIN.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_number": {"type": "string", "description": "The customer's account number."},
                "pin": {"type": "string", "description": "The customer's 4-digit security PIN."},
            },
            "required": ["account_number", "pin"],
        },
    },
    {
        "type": "function",
        "name": "get_bill_summary",
        "description": "Retrieve the current bill summary for an authenticated customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_number": {"type": "string", "description": "The customer's account number."}
            },
            "required": ["account_number"],
        },
    },
    {
        "type": "function",
        "name": "make_payment",
        "description": "Process a bill payment for an authenticated customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_number": {"type": "string", "description": "The customer's account number."},
                "amount": {"type": "number", "description": "Dollar amount to pay."},
                "payment_method": {
                    "type": "string",
                    "description": "Payment method: 'credit_card', 'debit_card', or 'bank_account'.",
                    "enum": ["credit_card", "debit_card", "bank_account"],
                },
            },
            "required": ["account_number", "amount", "payment_method"],
        },
    },
]

# ── System instructions ──────────────────────────────────────────────────────

SYSTEM_INSTRUCTIONS = """\
You are the automated voice assistant for PowerGrid Electric, a residential \
electric utility company. Your name is Sparky.

IMPORTANT: Always speak and respond in English only, regardless of the \
caller's language or accent.

Behaviour rules:
1. Greet the caller warmly and ask how you can help.
2. Before revealing any account details or processing payments, you MUST \
   authenticate the caller. Ask for their account number, then their 4-digit \
   PIN. Use the authenticate tool.
3. Once authenticated, you can look up their bill, read them the summary, \
   and process payments.
4. Always confirm payment details (amount, method) verbally before calling \
   make_payment.
5. Be concise, friendly, and professional. Keep responses brief — this is a \
   phone call, not an essay.
6. If the caller asks something outside billing / account / payment scope, \
   politely say you can only help with billing and payments today.
7. At the end, ask if there's anything else and wish them a good day.

Available test accounts for this POC demo:
- Account 12345 (PIN 9999)
- Account 67890 (PIN 1234)
"""

# ── Audio I/O ───────────────────────────────────────────────────────────────

audio_playback_queue: queue.Queue[bytes | None] = queue.Queue()

# True while the speaker is outputting audio.
# Set by playback worker, read by mic loop.
_speaker_lock = threading.Lock()
_speaker_playing = False


def _set_speaker(val: bool):
    global _speaker_playing
    with _speaker_lock:
        _speaker_playing = val


def _is_speaker_playing() -> bool:
    with _speaker_lock:
        return _speaker_playing


def playback_worker():
    pa = _open_pyaudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE,
        output=True, frames_per_buffer=FRAMES_PER_BUFFER,
    )
    try:
        while True:
            try:
                chunk = audio_playback_queue.get(timeout=0.5)
            except queue.Empty:
                # No audio for 500 ms — playback finished, unmute mic
                _set_speaker(False)
                chunk = audio_playback_queue.get()  # block until next chunk
            if chunk is None:
                break
            _set_speaker(True)
            stream.write(chunk)
        _set_speaker(False)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── WebSocket callbacks ─────────────────────────────────────────────────────

mic_stream = None
pa_instance = None
mic_thread_stop = threading.Event()


def send_session_update(ws):
    event = {
        "type": "session.update",
        "session": {
            "instructions": SYSTEM_INSTRUCTIONS,
            "tools": TOOLS,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
            "input_audio_noise_reduction": {"type": "far_field"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.9,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 600,
                "create_response": True,
                "interrupt_response": True,
            },
            "output_audio_format": "pcm16",
            "voice": "coral",
        },
    }
    ws.send(json.dumps(event))
    print("[session] Configuration sent.")


def start_mic_streaming(ws):
    global mic_stream, pa_instance

    pa_instance = _open_pyaudio()
    mic_stream = pa_instance.open(
        format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE,
        input=True, frames_per_buffer=FRAMES_PER_BUFFER,
    )

    def _mic_loop():
        if ECHO_MODE == "block":
            print("[mic] Streaming — mic muted while assistant speaks")
        else:
            print("[mic] Streaming — mic always open (echo handled elsewhere)")
        last_clear = 0.0

        while not mic_thread_stop.is_set():
            try:
                pcm = mic_stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)

                if ECHO_MODE == "block" and _is_speaker_playing():
                    now = time.monotonic()
                    if now - last_clear > 0.4:
                        ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                        last_clear = now
                    continue

                b64 = base64.b64encode(pcm).decode("utf-8")
                ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
            except Exception as e:
                if not mic_thread_stop.is_set():
                    print(f"[mic] Error: {e}")
                break

    threading.Thread(target=_mic_loop, daemon=True).start()


def handle_function_call(ws, call_id, name, arguments):
    print(f"[tool] Calling {name}({arguments})")
    func = TOOL_DISPATCH.get(name)
    if not func:
        result = {"error": f"Unknown function: {name}"}
    else:
        try:
            result = func(**json.loads(arguments))
        except Exception as e:
            result = {"error": str(e)}
    print(f"[tool] Result: {json.dumps(result)}")
    ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": json.dumps(result)},
    }))
    ws.send(json.dumps({"type": "response.create"}))


def on_open(ws):
    print("[ws] Connected to OpenAI Realtime API")
    send_session_update(ws)
    ws.send(json.dumps({"type": "response.create"}))
    start_mic_streaming(ws)
    threading.Thread(target=playback_worker, daemon=True).start()


def on_message(ws, message):
    event = json.loads(message)
    etype = event.get("type", "")

    if etype == "session.created":
        print(f"[session] Session created: {event.get('session', {}).get('id', 'N/A')}")

    elif etype == "session.updated":
        print("[session] Session updated successfully.")

    elif etype == "response.audio.delta":
        audio_b64 = event.get("delta", "")
        if audio_b64:
            audio_playback_queue.put(base64.b64decode(audio_b64))

    elif etype == "conversation.item.input_audio_transcription.completed":
        transcript = event.get("transcript", "").strip()
        if transcript:
            print(f"\n[you] {transcript}")

    elif etype == "conversation.item.input_audio_transcription.failed":
        print("\n[stt] Transcription failed for user audio.")

    elif etype == "response.audio_transcript.delta":
        text = event.get("delta", "")
        if text:
            print(text, end="", flush=True)

    elif etype == "response.audio_transcript.done":
        print()

    elif etype == "response.function_call_arguments.done":
        handle_function_call(ws, event.get("call_id", ""), event.get("name", ""), event.get("arguments", "{}"))

    elif etype == "input_audio_buffer.speech_started":
        if ECHO_MODE == "off":
            # Mic is open — user is interrupting.  Flush queued audio.
            print("\n[vad] User speech detected — interrupting playback")
            while True:
                try:
                    c = audio_playback_queue.get_nowait()
                    if c is None:
                        audio_playback_queue.put(None)
                        break
                except queue.Empty:
                    break
            _set_speaker(False)
            try:
                ws.send(json.dumps({"type": "response.cancel"}))
            except Exception:
                pass

    elif etype == "input_audio_buffer.speech_stopped":
        print("[vad] Speech ended — processing...")

    elif etype == "error":
        err = event.get("error", {})
        msg = err.get("message", "")
        if "no active response" not in msg:
            print(f"\n[error] {err.get('type', 'unknown')}: {msg}")

    elif etype in (
        "response.created", "response.done",
        "response.output_item.added", "response.output_item.done",
        "response.content_part.added", "response.content_part.done",
        "conversation.item.created", "input_audio_buffer.committed",
        "conversation.item.input_audio_transcription.delta", "rate_limits.updated",
    ):
        pass


def on_error(ws, error):
    print(f"\n[ws] Error: {error}")


def on_close(ws, close_status_code, close_msg):
    print(f"\n[ws] Disconnected (code={close_status_code})")
    mic_thread_stop.set()
    if mic_stream:
        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception:
            pass
    if pa_instance:
        try:
            pa_instance.terminate()
        except Exception:
            pass
    audio_playback_queue.put(None)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global ECHO_MODE

    parser = argparse.ArgumentParser(
        description="PowerGrid Electric Voice Assistant POC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
echo modes:
  block  Mute mic while assistant speaks. No barge-in.  (default)
  off    Mic always open. Use with headphones, telephony, or:
           pactl load-module module-echo-cancel""",
    )
    parser.add_argument("--echo", choices=["block", "off"], default="block",
                        help="Echo handling strategy (default: block)")
    args = parser.parse_args()
    ECHO_MODE = args.echo

    if not OPENAI_API_KEY:
        print("Error: Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    print("=" * 60)
    print("  PowerGrid Electric — Voice Assistant POC")
    print("  Model: gpt-realtime  |  Press Ctrl+C to quit")
    print(f"  Echo: {'mic blocked during playback' if ECHO_MODE == 'block' else 'mic always open'}")
    print("=" * 60)
    print()
    print("Demo accounts:")
    print("  Account 12345  PIN 9999  (John Smith)")
    print("  Account 67890  PIN 1234  (Jane Smith)")
    print()

    ws = websocket.WebSocketApp(
        WS_URL, header=WS_HEADERS,
        on_open=on_open, on_message=on_message,
        on_error=on_error, on_close=on_close,
    )
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
        mic_thread_stop.set()
        audio_playback_queue.put(None)
        ws.close()


if __name__ == "__main__":
    main()
