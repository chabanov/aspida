#!/bin/bash
# Example cURL commands for testing Aspida API

BASE_URL="${ASPIDA_BASE_URL:-http://localhost:8080}"
API_KEY="${ASPIDA_CLIENT_TOKEN:-test-token}"

echo "🤖 Aspida API Examples"
echo "   Base URL: $BASE_URL"
echo ""

# List models
echo "📋 List available models:"
echo "   curl -s $BASE_URL/v1/models | jq ."
curl -s "$BASE_URL/v1/models" | jq .
echo ""

# Single-turn chat (non-streaming)
echo "💬 Single-turn chat (non-streaming):"
echo "   curl -s -X POST $BASE_URL/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -H 'Authorization: Bearer $API_KEY' \\"
echo "     -d '{\"model\": \"qwen\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}]}' | jq ."
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "What is 2+2?"}]}' | jq .
echo ""

# Streaming chat
echo "🌊 Streaming chat:"
echo "   curl -N -X POST $BASE_URL/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -H 'Authorization: Bearer $API_KEY' \\"
echo "     -d '{\"model\": \"qwen\", \"messages\": [{\"role\": \"user\", \"content\": \"Count to 10\"}], \"stream\": true}'"
curl -N -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Count to 10"}], "stream": true}'
echo ""
echo ""

# Multi-turn conversation
echo "🗣️  Multi-turn conversation:"
echo "   curl -s -X POST $BASE_URL/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -H 'Authorization: Bearer $API_KEY' \\"
echo "     -d '{"
echo "       \"model\": \"qwen\","
echo "       \"messages\": ["
echo "         {\"role\": \"user\", \"content\": \"My name is Alice.\"},"
echo "         {\"role\": \"assistant\", \"content\": \"Nice to meet you, Alice!\"},"
echo "         {\"role\": \"user\", \"content\": \"What is my name?\"}"
echo "       ]"
echo "     }' | jq ."
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "qwen",
    "messages": [
      {"role": "user", "content": "My name is Alice."},
      {"role": "assistant", "content": "Nice to meet you, Alice!"},
      {"role": "user", "content": "What is my name?"}
    ]
  }' | jq .
echo ""

# With parameters
echo "⚙️  Chat with sampling parameters:"
echo "   curl -s -X POST $BASE_URL/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -H 'Authorization: Bearer $API_KEY' \\"
echo "     -d '{"
echo "       \"model\": \"qwen\","
echo "       \"messages\": [{\"role\": \"user\", \"content\": \"Write a haiku\"}],"
echo "       \"temperature\": 0.7,"
echo "       \"max_tokens\": 100"
echo "     }' | jq ."
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "temperature": 0.7,
    "max_tokens": 100
  }' | jq .
echo ""

echo "✅ Done!"
