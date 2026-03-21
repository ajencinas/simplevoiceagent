"""
Test script for outbound calls.

Usage:
    python test_outbound.py +15551234567                   # localhost
    python test_outbound.py +15551234567 https://xyz.ngrok-free.app  # ngrok
"""

import sys

from dotenv import load_dotenv
load_dotenv()

import requests

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_outbound.py <phone_number> [server_url]")
        print("  phone_number:  E.164 format, e.g. +15551234567")
        print("  server_url:    defaults to http://localhost:8000")
        sys.exit(1)

    phone_number = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    server_url = server_url.rstrip("/")

    url = f"{server_url}/outbound-call"
    payload = {"to": phone_number}

    print(f"Initiating outbound call to {phone_number}...")
    print(f"Server: {url}")

    try:
        resp = requests.post(url, json=payload, timeout=15)
        data = resp.json()

        if resp.status_code == 200:
            print(f"Call initiated!")
            print(f"  Call SID: {data.get('call_sid')}")
            print(f"  To:       {data.get('to')}")
            print(f"  Status:   {data.get('status')}")
        else:
            print(f"Error ({resp.status_code}): {data.get('error', data)}")

    except requests.ConnectionError:
        print(f"Could not connect to {server_url}. Is the server running?")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
