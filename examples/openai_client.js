#!/usr/bin/env node
/**
 * Example Node.js client for Aspida using OpenAI SDK.
 *
 * Usage:
 *   export ASPIDA_CLIENT_TOKEN=your-token
 *   node openai_client.js
 */

import OpenAI from 'openai';
import process from 'process';

const BASE_URL = process.env.ASPIDA_BASE_URL || 'http://localhost:8080/v1';
const API_KEY = process.env.ASPIDA_CLIENT_TOKEN || 'test-token';
const MODEL = process.env.ASPIDA_MODEL || 'qwen';

const client = new OpenAI({
  baseURL: BASE_URL,
  apiKey: API_KEY,
});

async function main() {
  console.log('🤖 Aspida Chat Client');
  console.log(`   Base URL: ${BASE_URL}`);
  console.log(`   Model: ${MODEL}`);
  console.log();

  // List available models
  console.log('📋 Available models:');
  const models = await client.models.list();
  for (const model of models.data) {
    console.log(`   - ${model.id}`);
  }
  console.log();

  // Single-turn chat
  console.log('💬 Single-turn chat:');
  const response = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: 'user', content: 'What is 2+2?' }],
    stream: false,
  });
  console.log(`   User: What is 2+2?`);
  console.log(`   Assistant: ${response.choices[0].message.content}`);
  console.log(`   Tokens: ${response.usage.total_tokens} (prompt: ${response.usage.prompt_tokens}, completion: ${response.usage.completion_tokens})`);
  console.log();

  // Streaming chat
  console.log('🌊 Streaming chat:');
  console.log(`   User: Tell me a short story about a robot.`);
  console.log(`   Assistant: `);

  const stream = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: 'user', content: 'Tell me a short story about a robot.' }],
    stream: true,
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
  }

  console.log('\n');

  // Multi-turn conversation
  console.log('🗣️  Multi-turn conversation:');
  const conversation = [
    { role: 'user', content: 'My name is Alice.' },
    { role: 'assistant', content: 'Nice to meet you, Alice!' },
    { role: 'user', content: "What's my name?" },
  ];

  const multiResponse = await client.chat.completions.create({
    model: MODEL,
    messages: conversation,
    stream: false,
  });

  console.log(`   User: ${conversation[0].content}`);
  console.log(`   Assistant: ${conversation[1].content}`);
  console.log(`   User: ${conversation[2].content}`);
  console.log(`   Assistant: ${multiResponse.choices[0].message.content}`);
  console.log();
}

main().catch(console.error);
