"""
Electric Utility Voice AI Assistant — POC
==========================================
Uses OpenAI Realtime API (gpt-realtime) over WebSocket for speech-to-speech
interaction.  Simulates authentication and bill-pay for a fictitious
electric utility ("PowerGrid Electric").

Echo handling (--echo flag):
    aec   — Acoustic echo cancellation via pyaec.  Mic stays open so the
             user can interrupt mid-sentence.  Best with speakers.  (default)
    gate  — Mute the mic while the assistant is speaking.  Simple, but
             blocks interruption.  Works without extra dependencies.
    off   — No echo handling.  Use when the transport already provides it
             (telephony / PBX, WebRTC, browser-based clients).

Requirements:
    pip install websocket-client pyaudio numpy
    pip install pyaec          # optional, for --echo aec (recommended)

Usage:
    export OPENAI_API_KEY="sk-..."
    python utility_voice_assistant.py                # AEC (default)
    python utility_voice_assistant.py --echo gate    # mic-gating
    python utility_voice_assistant.py --echo off     # telephony / WebRTC
"""

import argparse
import os
import sys
import json
import base64
import struct
import threading
import queue
import time
from datetime import date

import numpy as np
import pyaudio
import websocket

try:
    from pyaec import Aec as PyAec
    HAS_PYAEC = True
except ImportError:
    HAS_PYAEC = False

# ── Configuration ────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-realtime"
WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
WS_HEADERS = [
    f"Authorization: Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta: realtime=v1",
]
SAMPLE_RATE = 24_000          # API expects 24 kHz
CHANNELS = 1
FRAME_DURATION_MS = 100       # send audio every 100 ms
FRAMES_PER_BUFFER = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 2400 samples

# Echo-mode setting (set via CLI)
ECHO_MODE = "aec"              # "aec" | "gate" | "off"

# AEC parameters (only used when ECHO_MODE == "aec")
AEC_FILTER_LENGTH = 4800      # 200 ms tail — covers room reverb
ENERGY_GATE_THRESHOLD = 150   # RMS threshold after AEC to consider "real speech"
POST_AEC_HOLD_MS = 200        # short hold after last playback chunk to let AEC tail settle

# Gate parameters (only used when ECHO_MODE == "gate")
ASSISTANT_AUDIO_HOLD_SECONDS = 1.5

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

# Authenticated session state
session_state = {"authenticated": False, "account": None}

# ── Tool implementations (called when the model invokes a function) ──────────

def lookup_account(account_number: str) -> dict:
    """Look up an account by number."""
    acct = CUSTOMERS.get(account_number)
    if acct:
        return {"found": True, "name": acct["name"], "account_number": account_number}
    return {"found": False, "message": f"No account found for number {account_number}."}


def authenticate(account_number: str, pin: str) -> dict:
    """Verify customer identity with account number + PIN."""
    acct = CUSTOMERS.get(account_number)
    if not acct:
        return {"success": False, "message": "Account not found."}
    if acct["pin"] != pin:
        return {"success": False, "message": "Incorrect PIN. Please try again."}
    session_state["authenticated"] = True
    session_state["account"] = account_number
    return {"success": True, "message": f"Welcome, {acct['name']}! You are now authenticated."}


def get_bill_summary(account_number: str) -> dict:
    """Return current bill details (requires authentication)."""
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
    """Process a bill payment (simulated)."""
    if not session_state["authenticated"] or session_state["account"] != account_number:
        return {"error": "Please authenticate first."}
    acct = CUSTOMERS.get(account_number)
    if not acct:
        return {"error": "Account not found."}
    if amount <= 0:
        return {"error": "Payment amount must be positive."}
    if amount > acct["balance_due"]:
        return {"error": f"Amount ${amount:.2f} exceeds balance due of ${acct['balance_due']:.2f}."}

    # Simulate payment
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

# ── Tool definitions (sent to the model via session.update) ──────────────────

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
        "description": "Retrieve the current bill summary for an authenticated customer, including balance due, due date, usage, and last payment.",
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

# -- AEC reference buffer (shared between playback and mic threads) ----------

_ref_lock = threading.Lock()
_ref_buffer = bytearray()
_playback_active = False
_last_playback_time = 0.0


def _push_ref(pcm: bytes):
    """Record what the speaker is playing so AEC can subtract it."""
    global _playback_active, _last_playback_time
    with _ref_lock:
        _ref_buffer.extend(pcm)
        max_bytes = SAMPLE_RATE * 2  # keep at most 1 s
        if len(_ref_buffer) > max_bytes:
            del _ref_buffer[:len(_ref_buffer) - max_bytes]
        _playback_active = True
        _last_playback_time = time.monotonic()


def _pop_ref(n_bytes: int) -> bytes:
    """Pop reference audio aligned to the current mic frame."""
    with _ref_lock:
        available = min(n_bytes, len(_ref_buffer))
        if available == 0:
            return b'\x00' * n_bytes
        chunk = bytes(_ref_buffer[:available])
        del _ref_buffer[:available]
        if available < n_bytes:
            chunk += b'\x00' * (n_bytes - available)
        return chunk


def _is_playback_active() -> bool:
    with _ref_lock:
        if _playback_active:
            return True
        return (time.monotonic() - _last_playback_time) < (POST_AEC_HOLD_MS / 1000.0)


def _mark_playback_idle():
    global _playback_active
    with _ref_lock:
        _playback_active = False


# -- Gate state (used by "gate" mode) ----------------------------------------

assistant_audio_hold_until = 0.0


def mark_assistant_audio():
    global assistant_audio_hold_until
    assistant_audio_hold_until = time.monotonic() + ASSISTANT_AUDIO_HOLD_SECONDS


def assistant_audio_active() -> bool:
    return time.monotonic() < assistant_audio_hold_until


# -- Playback queue helpers --------------------------------------------------

def flush_playback_queue():
    """Drain pending audio from the playback queue (for interruption)."""
    flushed = 0
    while True:
        try:
            chunk = audio_playback_queue.get_nowait()
            if chunk is None:
                audio_playback_queue.put(None)  # preserve poison pill
                break
            flushed += 1
        except queue.Empty:
            break
    if flushed:
        print(f"[echo] Flushed {flushed} queued audio chunks (interruption)")
    with _ref_lock:
        _ref_buffer.clear()


def playback_worker():
    """Background thread that plays PCM16 audio from the queue."""
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
            chunk = audio_playback_queue.get()
            if chunk is None:
                break
            if ECHO_MODE == "aec":
                _push_ref(chunk)
            elif ECHO_MODE == "gate":
                mark_assistant_audio()
            stream.write(chunk)
        _mark_playback_idle()
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── Echo cancellation helpers (AEC mode) ────────────────────────────────────

def _create_echo_canceller():
    if not HAS_PYAEC:
        return None
    try:
        ec = PyAec(FRAMES_PER_BUFFER, AEC_FILTER_LENGTH, SAMPLE_RATE)
        print("[echo] pyaec adaptive echo canceller initialized")
        return ec
    except Exception as e:
        print(f"[echo] Failed to create pyaec canceller: {e}")
        return None


def _numpy_echo_cancel(mic_pcm: bytes, ref_pcm: bytes) -> bytes:
    """Fallback AEC using numpy spectral subtraction."""
    mic = np.frombuffer(mic_pcm, dtype=np.int16).astype(np.float32)
    ref = np.frombuffer(ref_pcm, dtype=np.int16).astype(np.float32)

    if np.max(np.abs(ref)) < 10:
        return mic_pcm  # reference is silence

    n_fft = len(mic)
    mic_fft = np.fft.rfft(mic, n=n_fft)
    ref_fft = np.fft.rfft(ref, n=n_fft)

    alpha = 1.5   # oversubtraction
    beta = 0.02   # spectral floor

    clean_mag = np.maximum(np.abs(mic_fft) - alpha * np.abs(ref_fft),
                           beta * np.abs(mic_fft))
    clean_fft = clean_mag * np.exp(1j * np.angle(mic_fft))
    clean = np.fft.irfft(clean_fft, n=n_fft)
    clean = np.clip(clean, -32768, 32767).astype(np.int16)
    return clean.tobytes()


def _rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


# ── WebSocket callbacks ─────────────────────────────────────────────────────

mic_stream = None
pa_instance = None
mic_thread_stop = threading.Event()


def send_session_update(ws):
    """Configure the session with tools and instructions."""
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
    """Capture microphone audio and stream to the API."""
    global mic_stream, pa_instance

    pa_instance = pyaudio.PyAudio()
    mic_stream = pa_instance.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    echo_canceller = _create_echo_canceller() if ECHO_MODE == "aec" else None

    def _mic_loop():
        mode_labels = {
            "aec": "echo cancellation active — speak any time",
            "gate": "mic-gating active — use headphones for best results",
            "off": "echo handling disabled (upstream provides it)",
        }
        print(f"[mic] Streaming started — {mode_labels[ECHO_MODE]}")
        frame_bytes = FRAMES_PER_BUFFER * 2
        last_clear_sent = 0.0

        while not mic_thread_stop.is_set():
            try:
                mic_pcm = mic_stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)

                # ── gate mode: mute mic while assistant is speaking ──
                if ECHO_MODE == "gate":
                    if assistant_audio_active():
                        now = time.monotonic()
                        if now - last_clear_sent > 0.4:
                            ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                            last_clear_sent = now
                        continue
                    send_pcm = mic_pcm

                # ── aec mode: echo-cancel and energy-gate ──
                elif ECHO_MODE == "aec":
                    ref_pcm = _pop_ref(frame_bytes)
                    if echo_canceller is not None:
                        try:
                            clean_samples = echo_canceller.cancel_echo(mic_pcm, ref_pcm)
                            send_pcm = struct.pack(f'<{len(clean_samples)}h', *clean_samples)
                        except Exception:
                            send_pcm = _numpy_echo_cancel(mic_pcm, ref_pcm)
                    else:
                        send_pcm = _numpy_echo_cancel(mic_pcm, ref_pcm)

                    if _is_playback_active():
                        if _rms(send_pcm) < ENERGY_GATE_THRESHOLD:
                            continue

                # ── off mode: raw passthrough ──
                else:
                    send_pcm = mic_pcm

                b64_audio = base64.b64encode(send_pcm).decode("utf-8")
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64_audio,
                }))
            except Exception as e:
                if not mic_thread_stop.is_set():
                    print(f"[mic] Error: {e}")
                break

    t = threading.Thread(target=_mic_loop, daemon=True)
    t.start()


def handle_function_call(ws, call_id: str, name: str, arguments: str):
    """Execute a tool call and send the result back."""
    print(f"[tool] Calling {name}({arguments})")
    func = TOOL_DISPATCH.get(name)
    if not func:
        result = {"error": f"Unknown function: {name}"}
    else:
        try:
            args = json.loads(arguments)
            result = func(**args)
        except Exception as e:
            result = {"error": str(e)}

    print(f"[tool] Result: {json.dumps(result)}")

    ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result),
        },
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
            pcm = base64.b64decode(audio_b64)
            audio_playback_queue.put(pcm)

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
        call_id = event.get("call_id", "")
        name = event.get("name", "")
        arguments = event.get("arguments", "{}")
        handle_function_call(ws, call_id, name, arguments)

    elif etype == "input_audio_buffer.speech_started":
        if ECHO_MODE != "gate":
            # In AEC/off modes, flush queued audio for snappy interruption
            print("[echo] User speech detected — interrupting playback")
            flush_playback_queue()
            _mark_playback_idle()

    elif etype == "input_audio_buffer.speech_stopped":
        print("[vad] Speech ended — processing...")

    elif etype == "response.done":
        _mark_playback_idle()

    elif etype == "error":
        err = event.get("error", {})
        print(f"\n[error] {err.get('type', 'unknown')}: {err.get('message', '')}")

    elif etype in (
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "conversation.item.created",
        "input_audio_buffer.committed",
        "conversation.item.input_audio_transcription.delta",
        "rate_limits.updated",
    ):
        pass

    else:
        # Uncomment for debugging:
        # print(f"[event] {etype}")
        pass


def on_error(ws, error):
    print(f"\n[ws] Error: {error}")


def on_close(ws, close_status_code, close_msg):
    print(f"\n[ws] Disconnected (code={close_status_code})")
    mic_thread_stop.set()
    if mic_stream is not None:
        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception:
            pass
    if pa_instance is not None:
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
  aec    Acoustic echo cancellation (pyaec). Allows interruption.  (default)
  gate   Mute mic during playback. Simple but blocks interruption.
  off    No echo handling. For telephony/PBX/WebRTC integrations.""",
    )
    parser.add_argument(
        "--echo",
        choices=["aec", "gate", "off"],
        default="aec",
        help="Echo handling strategy (default: aec)",
    )
    args = parser.parse_args()
    ECHO_MODE = args.echo

    if not OPENAI_API_KEY:
        print("Error: Set the OPENAI_API_KEY environment variable.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    echo_labels = {
        "aec": "pyaec (adaptive filter)" if HAS_PYAEC else "numpy spectral subtraction (fallback)",
        "gate": "mic-gating (1.5 s hold)",
        "off": "DISABLED (upstream handles echo)",
    }

    print("=" * 60)
    print("  PowerGrid Electric — Voice Assistant POC")
    print("  Model: gpt-realtime  |  Press Ctrl+C to quit")
    print(f"  Echo mode: {echo_labels[ECHO_MODE]}")
    if ECHO_MODE == "aec" and not HAS_PYAEC:
        print("  Tip: pip install pyaec for better echo cancellation")
    if ECHO_MODE == "gate":
        print("  Tip: Use headphones to avoid echo issues")
    print("=" * 60)
    print()
    print("Demo accounts:")
    print("  Account 12345  PIN 9999  (John Smith)")
    print("  Account 67890  PIN 1234  (Jane Smith)")
    print()

    ws = websocket.WebSocketApp(
        WS_URL,
        header=WS_HEADERS,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
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
