# PowerGrid Electric Voice AI Assistant

A Proof of Concept (POC) demonstrating an automated voice assistant for an electric utility company. An AI assistant named **Sparky** handles customer calls using OpenAI's Realtime API for real-time speech-to-speech interaction.

The project supports two modes:

1. **Phone calls via Twilio** — a FastAPI server (`api_server.py`) bridges Twilio Voice to the OpenAI Realtime API, handling both inbound and outbound calls.
2. **Local microphone** — a standalone script (`utility_voice_assistant.py`) talks to the assistant directly from your computer's mic and speakers.

## Features

- **Account Lookup** – Verify account existence by account number
- **Customer Authentication** – Authenticate using account number + 4-digit PIN
- **Bill Inquiry** – Retrieve balance due, due date, energy usage, plan type, and payment history
- **Payment Processing** – Process payments via credit card, debit card, or bank account
- **Inbound & Outbound Calls** – Receive calls to a Twilio number or place outbound calls programmatically

## Tech Stack

- **OpenAI Realtime API** (`gpt-realtime`) – Speech-to-speech interaction over WebSocket (server VAD, tool calls, voice `coral`)
- **Twilio Voice** – Inbound/outbound PSTN calls with Media Streams (mulaw 8 kHz)
- **FastAPI + Uvicorn** – HTTP/WebSocket server bridging Twilio ↔ OpenAI
- **PyAudio / numpy** – Real-time audio capture, playback, and conversion for local/test clients

## Architecture

`api_server.py` runs a FastAPI app on port `8000` that bridges Twilio and OpenAI:

| Endpoint | Type | Purpose |
|----------|------|---------|
| `/incoming-call` | POST | Twilio webhook for inbound calls → returns `<Stream>` TwiML |
| `/outbound-call` | POST | Initiates an outbound call via the Twilio API |
| `/outbound-call-twiml` | POST | TwiML for outbound calls → returns `<Stream>` |
| `/twilio-ws` | WebSocket | Shared media-stream handler for inbound & outbound calls |
| `/ws` | WebSocket | Direct audio bridge for local/test clients (PCM16 24 kHz) |
| `/health` | GET | Health check |

Audio is converted between Twilio's mulaw 8 kHz and OpenAI's PCM16 24 kHz on the fly. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full diagram and call flows.

## Project Structure

```
aiassist/
├── api_server.py                # FastAPI server: Twilio ↔ OpenAI Realtime bridge
├── utility_voice_assistant.py   # Standalone local mic/speaker assistant
├── test_client.py               # Test client: mic + speaker over /ws
├── test_client_text.py          # Test client: text input + speaker over /ws
├── test_outbound.py             # Initiate an outbound call via /outbound-call
├── test_utility_assistant.py    # Automated conversation harness (no mic required)
├── ARCHITECTURE.md              # Architecture diagram and call flows
├── requirements.txt
└── .gitignore
```

## Prerequisites

- Python 3.12+
- An OpenAI API key with access to the Realtime API
- A Twilio account with a voice-capable phone number (for phone calls)
- [ngrok](https://ngrok.com/) (or equivalent) to expose the local server to Twilio
- Audio input/output devices for the local/test clients (microphone + speakers/headphones)

> **Tip:** Headphones are recommended to avoid echo from VAD sensitivity when using speakers.

## Setup

```bash
# Create and activate a virtual environment
python -m venv aiassist
source aiassist/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root (this file is gitignored — never commit secrets):

```bash
OPENAI_API_KEY=sk-...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
BASE_URL=https://your-subdomain.ngrok-free.dev   # public URL of this server
```

## Usage

### Phone calls (Twilio)

1. **Start the server:**

   ```bash
   source aiassist/bin/activate
   python api_server.py
   ```

2. **Expose it publicly** so Twilio can reach the media stream (must match `BASE_URL`):

   ```bash
   ngrok http --url=your-subdomain.ngrok-free.dev 8000
   ```

3. **Inbound:** point your Twilio number's Voice webhook to
   `https://your-subdomain.ngrok-free.dev/incoming-call`, then call the number.

4. **Outbound:** trigger a call to any phone number:

   ```bash
   python test_outbound.py +15551234567 https://your-subdomain.ngrok-free.dev
   ```

### Local clients (no phone)

With the server running, connect directly over the `/ws` endpoint:

```bash
python test_client.py        # microphone + speaker
python test_client_text.py   # typed input + spoken responses
```

### Standalone local assistant

Run the interactive assistant directly against the OpenAI Realtime API (no server, no Twilio):

```bash
python utility_voice_assistant.py
```

### Automated test

Run the test harness that simulates a full conversation without a microphone:

```bash
python test_utility_assistant.py
```

The test walks through a complete flow: greeting, account lookup, PIN authentication, bill inquiry, payment, and farewell.

## Test Accounts

| Account | PIN  | Customer   | Balance  | Due Date   |
|---------|------|------------|----------|------------|
| 12345   | 9999 | John Smith | $147.83  | 2026-03-15 |
| 67890   | 1234 | Jane Smith | $203.45  | 2026-03-20 |

## Audio Configuration

| Path | Sample Rate | Format |
|------|-------------|--------|
| OpenAI Realtime / local clients | 24 kHz | PCM16 (mono) |
| Twilio Media Streams | 8 kHz | mulaw (mono) |

- **Frame Duration:** 100 ms chunks
- **Voice Activity Detection:** Server-side with configurable thresholds
