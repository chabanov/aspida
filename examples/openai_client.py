#!/usr/bin/env python3
"""
Example Python client for Aspida using OpenAI SDK.

Usage:
    export ASPIDA_CLIENT_TOKEN=your-token
    python openai_client.py
"""

import os
from openai import OpenAI

# Configuration
BASE_URL = os.getenv("ASPIDA_BASE_URL", "http://localhost:8080/v1")
API_KEY = os.getenv("ASPIDA_CLIENT_TOKEN", "test-token")
MODEL = os.getenv("ASPIDA_MODEL", "qwen")


def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    print("🤖 Aspida Chat Client")
    print(f"   Base URL: {BASE_URL}")
    print(f"   Model: {MODEL}")
    print()

    # List available models
    print("📋 Available models:")
    models = client.models.list()
    for model in models.data:
        print(f"   - {model.id}")
    print()

    # Single-turn chat
    print("💬 Single-turn chat:")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        stream=False
    )
    print(f"   User: What is 2+2?")
    print(f"   Assistant: {response.choices[0].message.content}")
    print(f"   Tokens: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
    print()

    # Streaming chat
    print("🌊 Streaming chat:")
    print(f"   User: Tell me a short story about a robot.")
    print(f"   Assistant: ", end="", flush=True)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a short story about a robot."}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")

    # Multi-turn conversation
    print("🗣️  Multi-turn conversation:")
    conversation = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What's my name?"}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=conversation,
        stream=False
    )

    print(f"   User: {conversation[0]['content']}")
    print(f"   Assistant: {conversation[1]['content']}")
    print(f"   User: {conversation[2]['content']}")
    print(f"   Assistant: {response.choices[0].message.content}")
    print()


if __name__ == "__main__":
    main()
