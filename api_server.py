"""
PowerGrid Electric — WebSocket API Server
==========================================
Bridges remote clients to the OpenAI Realtime API.
Supports inbound Twilio calls, outbound Twilio calls, and direct WebSocket clients.

Client connects via WebSocket, sends/receives JSON messages:
  → {"type": "audio", "audio": "<base64 PCM16 24kHz mono>"}
  → {"type": "text", "text": "message"}
  ← {"type": "audio", "audio": "<base64 PCM16 24kHz mono>"}
  ← {"type": "transcript", "role": "user", "text": "..."}
  ← {"type": "transcript", "role": "assistant", "text": "..."}
  ← {"type": "error", "message": "..."}

Outbound calls:
  POST /outbound-call {"to": "+15551234567"}

Usage:
    export OPENAI_API_KEY="sk-..."
    export TWILIO_ACCOUNT_SID="AC..."
    export TWILIO_AUTH_TOKEN="..."
    export TWILIO_PHONE_NUMBER="+1..."
    export BASE_URL="https://xyz.ngrok-free.app"
    python api_server.py                    # starts on port 8000
    ngrok http 8000                         # expose publicly
"""

import asyncio
import audioop
import base64
import json
import os
import struct
import sys
import time
from datetime import date

from dotenv import load_dotenv
load_dotenv()

import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from twilio.rest import Client as TwilioClient
import uvicorn

# ── Configuration ────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-realtime"
WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
SAMPLE_RATE = 24_000

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
BASE_URL = os.environ.get("BASE_URL", "")  # e.g. https://xyz.ngrok-free.app

app = FastAPI(title="PowerGrid Electric Voice API")

# ── Fake customer database (same as voice assistant) ─────────────────────────

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

# ── Tool implementations (per-session state) ────────────────────────────────

def lookup_account(account_number: str) -> dict:
    acct = CUSTOMERS.get(account_number)
    if acct:
        return {"found": True, "name": acct["name"], "account_number": account_number}
    return {"found": False, "message": f"No account found for number {account_number}."}


def authenticate(session_state: dict, account_number: str, pin: str) -> dict:
    acct = CUSTOMERS.get(account_number)
    if not acct:
        return {"success": False, "message": "Account not found."}
    if acct["pin"] != pin:
        return {"success": False, "message": "Incorrect PIN. Please try again."}
    session_state["authenticated"] = True
    session_state["account"] = account_number
    return {"success": True, "message": f"Welcome, {acct['name']}! You are now authenticated."}


def get_bill_summary(session_state: dict, account_number: str) -> dict:
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


def make_payment(session_state: dict, account_number: str, amount: float, payment_method: str) -> dict:
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


# ── Tool / session config (reused from original) ────────────────────────────

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


# ── Tool dispatch helper ─────────────────────────────────────────────────────

def dispatch_tool(session_state: dict, name: str, arguments: str) -> str:
    args = json.loads(arguments)
    if name == "lookup_account":
        result = lookup_account(**args)
    elif name == "authenticate":
        result = authenticate(session_state, **args)
    elif name == "get_bill_summary":
        result = get_bill_summary(session_state, **args)
    elif name == "make_payment":
        result = make_payment(session_state, **args)
    else:
        result = {"error": f"Unknown function: {name}"}
    return json.dumps(result)


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    print("[server] Client connected")

    session_state = {"authenticated": False, "account": None}

    # Connect to OpenAI Realtime API
    extra_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        async with websockets.connect(WS_URL, additional_headers=extra_headers) as openai_ws:
            # Send session configuration
            await openai_ws.send(json.dumps({
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
            }))
            print("[server] Session config sent to OpenAI")

            # Trigger initial greeting
            await openai_ws.send(json.dumps({"type": "response.create"}))

            # Two concurrent tasks: client→OpenAI and OpenAI→client
            async def client_to_openai():
                """Forward client messages to OpenAI."""
                try:
                    while True:
                        raw = await client_ws.receive_text()
                        msg = json.loads(raw)
                        msg_type = msg.get("type", "")

                        if msg_type == "audio" and msg.get("audio"):
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": msg["audio"],
                            }))

                        elif msg_type == "commit":
                            silence = base64.b64encode(b"\x00" * (SAMPLE_RATE // 2 * 2)).decode()
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": silence,
                            }))
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.commit",
                            }))
                            await openai_ws.send(json.dumps({
                                "type": "response.create",
                            }))
                            print("[server] Audio committed + response requested")

                        elif msg_type == "text" and msg.get("text"):
                            await openai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{
                                        "type": "input_text",
                                        "text": msg["text"],
                                    }],
                                },
                            }))
                            await openai_ws.send(json.dumps({
                                "type": "response.create",
                            }))
                            print(f"[server] Text forwarded: {msg['text']}")

                except WebSocketDisconnect:
                    print("[server] Client disconnected")
                except Exception as e:
                    print(f"[server] client_to_openai error: {e}")

            async def openai_to_client():
                """Forward OpenAI responses to client."""
                try:
                    async for raw in openai_ws:
                        event = json.loads(raw)
                        etype = event.get("type", "")

                        if etype == "response.audio.delta":
                            audio_b64 = event.get("delta", "")
                            if audio_b64:
                                await client_ws.send_json({
                                    "type": "audio",
                                    "audio": audio_b64,
                                })

                        elif etype == "conversation.item.input_audio_transcription.completed":
                            transcript = event.get("transcript", "").strip()
                            if transcript:
                                await client_ws.send_json({
                                    "type": "transcript",
                                    "role": "user",
                                    "text": transcript,
                                })
                                print(f"[server] User said: {transcript}")

                        elif etype == "response.audio_transcript.done":
                            transcript = event.get("transcript", "").strip()
                            if transcript:
                                await client_ws.send_json({
                                    "type": "transcript",
                                    "role": "assistant",
                                    "text": transcript,
                                })
                                print(f"[server] Assistant said: {transcript}")

                        elif etype == "response.function_call_arguments.done":
                            call_id = event.get("call_id", "")
                            name = event.get("name", "")
                            arguments = event.get("arguments", "{}")
                            print(f"[server] Tool call: {name}({arguments})")
                            result = dispatch_tool(session_state, name, arguments)
                            print(f"[server] Tool result: {result}")
                            await openai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result,
                                },
                            }))
                            await openai_ws.send(json.dumps({"type": "response.create"}))

                        elif etype == "error":
                            err = event.get("error", {})
                            emsg = err.get("message", "")
                            print(f"[server] ERROR: {err.get('type', 'unknown')}: {emsg}")
                            await client_ws.send_json({
                                "type": "error",
                                "message": emsg,
                            })

                except Exception as e:
                    print(f"[server] openai_to_client error: {e}")

            # Run both directions concurrently
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_openai()),
                    asyncio.create_task(openai_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except Exception as e:
        print(f"[server] Connection error: {e}")
        try:
            await client_ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

    print("[server] Session ended")


# ── Audio conversion helpers (Twilio ↔ OpenAI) ──────────────────────────────

def mulaw_8k_to_pcm16_24k(mulaw_bytes: bytes) -> bytes:
    """Convert mulaw 8kHz to PCM16 24kHz (what OpenAI expects)."""
    # mulaw → PCM16 at 8kHz
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    # 8kHz → 24kHz (3x upsample)
    pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
    return pcm_24k


def pcm16_24k_to_mulaw_8k(pcm_bytes: bytes) -> bytes:
    """Convert PCM16 24kHz to mulaw 8kHz (what Twilio expects)."""
    # 24kHz → 8kHz (3x downsample)
    pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, 24000, 8000, None)
    # PCM16 → mulaw
    mulaw = audioop.lin2ulaw(pcm_8k, 2)
    return mulaw


# ── Twilio TwiML endpoint ───────────────────────────────────────────────────

@app.post("/incoming-call")
async def incoming_call(request: Request):
    """Returns TwiML that tells Twilio to open a Media Stream WebSocket."""
    # Build the WebSocket URL from the request
    host = request.headers.get("host", "localhost:8000")
    scheme = "wss" if request.headers.get("x-forwarded-proto") == "https" else "ws"
    ws_url = f"{scheme}://{host}/twilio-ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""

    print(f"[twilio] Incoming call → streaming to {ws_url}")
    return Response(content=twiml, media_type="application/xml")


# ── Twilio Media Stream WebSocket ────────────────────────────────────────────

@app.websocket("/twilio-ws")
async def twilio_websocket_endpoint(twilio_ws: WebSocket):
    await twilio_ws.accept()
    print("[twilio] Media stream connected")

    session_state = {"authenticated": False, "account": None}
    stream_sid = None

    extra_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        async with websockets.connect(WS_URL, additional_headers=extra_headers) as openai_ws:
            # Configure OpenAI session — same as /ws but with g711_ulaw input
            await openai_ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": SYSTEM_INSTRUCTIONS,
                    "tools": TOOLS,
                    "modalities": ["audio", "text"],
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
                    "input_audio_noise_reduction": {"type": "near_field"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                    "output_audio_format": "pcm16",
                    "voice": "coral",
                },
            }))
            print("[twilio] Session config sent to OpenAI")

            # Trigger initial greeting
            await openai_ws.send(json.dumps({"type": "response.create"}))

            async def twilio_to_openai():
                """Forward Twilio media to OpenAI."""
                try:
                    while True:
                        raw = await twilio_ws.receive_text()
                        msg = json.loads(raw)
                        event = msg.get("event", "")

                        if event == "start":
                            nonlocal stream_sid
                            stream_sid = msg.get("start", {}).get("streamSid")
                            print(f"[twilio] Stream started: {stream_sid}")

                        elif event == "media":
                            payload = msg.get("media", {}).get("payload", "")
                            if payload:
                                # Decode mulaw from Twilio, convert to PCM16 24kHz
                                mulaw_bytes = base64.b64decode(payload)
                                pcm_24k = mulaw_8k_to_pcm16_24k(mulaw_bytes)
                                b64_pcm = base64.b64encode(pcm_24k).decode()
                                await openai_ws.send(json.dumps({
                                    "type": "input_audio_buffer.append",
                                    "audio": b64_pcm,
                                }))

                        elif event == "stop":
                            print("[twilio] Stream stopped")
                            break

                except WebSocketDisconnect:
                    print("[twilio] Twilio disconnected")
                except Exception as e:
                    print(f"[twilio] twilio_to_openai error: {e}")

            async def openai_to_twilio():
                """Forward OpenAI audio to Twilio."""
                try:
                    async for raw in openai_ws:
                        event = json.loads(raw)
                        etype = event.get("type", "")

                        if etype == "response.audio.delta":
                            audio_b64 = event.get("delta", "")
                            if audio_b64 and stream_sid:
                                # Convert PCM16 24kHz → mulaw 8kHz
                                pcm_bytes = base64.b64decode(audio_b64)
                                mulaw_bytes = pcm16_24k_to_mulaw_8k(pcm_bytes)
                                mulaw_b64 = base64.b64encode(mulaw_bytes).decode()
                                await twilio_ws.send_json({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": mulaw_b64,
                                    },
                                })

                        elif etype == "response.audio_transcript.done":
                            transcript = event.get("transcript", "").strip()
                            if transcript:
                                print(f"[twilio] Assistant said: {transcript}")

                        elif etype == "conversation.item.input_audio_transcription.completed":
                            transcript = event.get("transcript", "").strip()
                            if transcript:
                                print(f"[twilio] Caller said: {transcript}")

                        elif etype == "response.function_call_arguments.done":
                            call_id = event.get("call_id", "")
                            name = event.get("name", "")
                            arguments = event.get("arguments", "{}")
                            print(f"[twilio] Tool call: {name}({arguments})")
                            result = dispatch_tool(session_state, name, arguments)
                            print(f"[twilio] Tool result: {result}")
                            await openai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result,
                                },
                            }))
                            await openai_ws.send(json.dumps({"type": "response.create"}))

                        elif etype == "input_audio_buffer.speech_started":
                            # User is interrupting — clear Twilio's audio buffer
                            if stream_sid:
                                await twilio_ws.send_json({
                                    "event": "clear",
                                    "streamSid": stream_sid,
                                })
                                print("[twilio] Barge-in — cleared Twilio audio buffer")

                        elif etype == "error":
                            err = event.get("error", {})
                            emsg = err.get("message", "")
                            print(f"[twilio] ERROR: {err.get('type', 'unknown')}: {emsg}")

                except Exception as e:
                    print(f"[twilio] openai_to_twilio error: {e}")

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(twilio_to_openai()),
                    asyncio.create_task(openai_to_twilio()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except Exception as e:
        print(f"[twilio] Connection error: {e}")

    print("[twilio] Session ended")


# ── Outbound call endpoints ──────────────────────────────────────────────────

@app.post("/outbound-call")
async def outbound_call(request: Request):
    """Initiate an outbound call via Twilio."""
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, BASE_URL]):
        return JSONResponse(
            status_code=500,
            content={"error": "Missing Twilio configuration. Set TWILIO_ACCOUNT_SID, "
                     "TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, and BASE_URL env vars."},
        )

    body = await request.json()
    to_number = body.get("to", "").strip()
    if not to_number:
        return JSONResponse(status_code=400, content={"error": "Missing 'to' phone number."})

    try:
        twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{BASE_URL}/outbound-call-twiml",
        )
        print(f"[outbound] Call initiated to {to_number} — SID: {call.sid}")
        return {"status": "calling", "call_sid": call.sid, "to": to_number}
    except Exception as e:
        print(f"[outbound] Error initiating call: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/outbound-call-twiml")
async def outbound_call_twiml(request: Request):
    """Returns TwiML that opens a media stream for an outbound call."""
    host = request.headers.get("host", "localhost:8000")
    scheme = "wss" if request.headers.get("x-forwarded-proto") == "https" else "ws"
    ws_url = f"{scheme}://{host}/twilio-ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""

    print(f"[outbound] Call connected → streaming to {ws_url}")
    return Response(content=twiml, media_type="application/xml")


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "PowerGrid Electric Voice API"}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Error: Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    print("=" * 60)
    print("  PowerGrid Electric — Voice API Server")
    print("  WebSocket endpoint:  ws://localhost:8000/ws")
    print("  Twilio inbound:      http://localhost:8000/incoming-call")
    print("  Twilio outbound:     POST http://localhost:8000/outbound-call")
    print("  Twilio WebSocket:    ws://localhost:8000/twilio-ws")
    print("  Health check:        http://localhost:8000/health")
    if BASE_URL:
        print(f"  Base URL:            {BASE_URL}")
    else:
        print("  ⚠ BASE_URL not set — outbound calls won't work")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
