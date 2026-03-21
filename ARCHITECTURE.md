# PowerGrid Electric — Voice API Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         api_server.py                               │
│                    (FastAPI + Uvicorn :8000)                         │
│                                                                     │
│  ┌───────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  /ws           │  │  /incoming-call   │  │  /outbound-call      │  │
│  │  (WebSocket)   │  │  (POST → TwiML)  │  │  (POST → Twilio API) │  │
│  └──────┬────────┘  └───────┬──────────┘  └──────────┬───────────┘  │
│         │                   │                        │              │
│         │                   ▼                        ▼              │
│         │           ┌──────────────┐        ┌──────────────────┐    │
│         │           │ /twilio-ws   │        │/outbound-call-   │    │
│         │           │ (WebSocket)  │        │ twiml (POST)     │    │
│         │           └──────┬───────┘        └───────┬──────────┘    │
│         │                  │                        │               │
│         │                  │    ┌────────────────────┘               │
│         │                  │    │  (returns <Stream> TwiML)          │
│         │                  ▼    ▼                                    │
│         │           ┌──────────────┐                                │
│         │           │  /twilio-ws  │◄── shared handler for both     │
│         │           │              │    inbound & outbound calls     │
│         │           └──────┬───────┘                                │
│         │                  │                                        │
│         ▼                  ▼                                        │
│  ┌─────────────────────────────────────────┐                        │
│  │         OpenAI Realtime API Bridge       │                        │
│  │  ┌─────────────────────────────────┐    │                        │
│  │  │  Audio conversion (mulaw↔PCM16) │    │                        │
│  │  │  Tool dispatch (dispatch_tool)  │    │                        │
│  │  │  Session state management       │    │                        │
│  │  └─────────────────────────────────┘    │                        │
│  └─────────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘

                    EXTERNAL SERVICES
┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  Twilio       │  │  OpenAI Realtime │  │  Test Clients        │
│  Voice API    │  │  API (WebSocket) │  │                      │
│               │  │                  │  │  test_client.py      │
│  • Inbound    │  │  • gpt-realtime  │  │   (mic + speaker)    │
│    calls      │  │  • server VAD    │  │                      │
│  • Outbound   │  │  • PCM16 24kHz   │  │  test_client_text.py │
│    calls      │  │  • Tool calls    │  │   (text + speaker)   │
│  • mulaw 8kHz │  │  • Voice: coral  │  │                      │
│               │  │                  │  │  test_outbound.py    │
└──────┬───────┘  └────────┬─────────┘  │   (initiate calls)   │
       │                   │            └──────────┬───────────┘
       │                   │                       │
       ▼                   ▼                       ▼

═══════════════════════════════════════════════════════════════════

## Call Flows

### Inbound Call
  Caller ──► Twilio ──► POST /incoming-call
                            │
                            ▼ (TwiML: <Stream>)
                        /twilio-ws ◄──► OpenAI Realtime API
                            │
                            ▼ (mulaw ↔ PCM16 conversion)
                        Sparky talks to caller

### Outbound Call
  test_outbound.py ──► POST /outbound-call
                            │
                            ▼ (Twilio REST API)
                        Twilio dials phone number
                            │
                            ▼ (person picks up)
                        Twilio fetches POST /outbound-call-twiml
                            │
                            ▼ (TwiML: <Stream>)
                        /twilio-ws ◄──► OpenAI Realtime API
                            │
                            ▼ (mulaw ↔ PCM16 conversion)
                        Sparky talks to callee

### Direct WebSocket Client
  test_client.py ──► ws://host:8000/ws ◄──► OpenAI Realtime API
                     (PCM16 24kHz both directions, no conversion)

═══════════════════════════════════════════════════════════════════

## Tools Available to Sparky (AI Assistant)

  ┌──────────────────┐     ┌───────────────┐
  │  lookup_account  │     │  authenticate │
  │  (account_number)│     │  (acct + PIN) │
  └──────────────────┘     └───────────────┘
  ┌──────────────────┐     ┌───────────────┐
  │ get_bill_summary │     │ make_payment  │
  │ (needs auth)     │     │ (needs auth)  │
  └──────────────────┘     └───────────────┘

═══════════════════════════════════════════════════════════════════

## Environment Variables

  OPENAI_API_KEY        → OpenAI Realtime API access
  TWILIO_ACCOUNT_SID    → Twilio account
  TWILIO_AUTH_TOKEN     → Twilio auth
  TWILIO_PHONE_NUMBER   → Twilio outbound caller ID
  BASE_URL              → Public URL (ngrok) for Twilio callbacks
```
