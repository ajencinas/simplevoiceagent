# PowerGrid Electric Voice AI Assistant

A Proof of Concept (POC) demonstrating an automated voice assistant for an electric utility company. It uses OpenAI's Realtime API over WebSocket to enable real-time speech-to-speech interaction with an AI assistant named **Sparky**.

## Features

- **Account Lookup** – Verify account existence by account number
- **Customer Authentication** – Authenticate using account number + 4-digit PIN
- **Bill Inquiry** – Retrieve balance due, due date, energy usage, plan type, and payment history
- **Payment Processing** – Process payments via credit card, debit card, or bank account

## Tech Stack

- **OpenAI Realtime API** (`gpt-realtime`) – Speech-to-speech interaction over WebSocket
- **OpenAI TTS** (`gpt-4o-mini-tts`) – Text-to-speech for test audio generation
- **PyAudio** – Real-time microphone capture and speaker playback
- **websocket-client** – WebSocket communication
- **numpy** – Audio processing

## Project Structure

```
aiassist/
├── utility_voice_assistant.py   # Main live voice assistant
├── test_utility_assistant.py    # Automated test harness (no mic required)
└── .gitignore
```

## Prerequisites

- Python 3.12+
- An OpenAI API key with access to the Realtime API
- Audio input/output devices (microphone + speakers/headphones)

> **Tip:** Headphones are recommended to avoid echo from VAD sensitivity when using speakers.

## Setup

```bash
# Create and activate a virtual environment
python -m venv aiassist
source aiassist/bin/activate

# Install dependencies
pip install websocket-client pyaudio numpy openai pydantic httpx

# Set your API key
export OPENAI_API_KEY="sk-..."
```

## Usage

### Live Voice Assistant

Run the interactive assistant with microphone input:

```bash
python utility_voice_assistant.py
```

### Automated Test

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

- **Sample Rate:** 24 kHz
- **Format:** PCM16 (mono)
- **Frame Duration:** 100 ms chunks
- **Voice Activity Detection:** Server-side with configurable thresholds
