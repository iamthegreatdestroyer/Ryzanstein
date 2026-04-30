/**
 * Ryzanstein LLM API - Standard Load Test
 *
 * Purpose: Ramp load gradually to realistic operational levels
 * Load Profile: 10 → 25 → 50 VUs over 5 minutes
 * Duration: 5 minutes
 * Focus: Sustained performance under increasing load
 */

import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 10 },  // Ramp up to 10 VUs
    { duration: '2m', target: 25 },  // Ramp up to 25 VUs
    { duration: '2m', target: 50 },  // Ramp up to 50 VUs
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000', 'p(99)<2000'],
    http_req_failed: ['rate<0.05'],
  },
};

const API_URL = __ENV.API_URL || 'http://localhost:8000';

export default function () {
  // 70% chat completions (heavy workload)
  if (Math.random() < 0.7) {
    let chatRes = http.post(`${API_URL}/v1/chat/completions`, JSON.stringify({
      model: 'ryzanstein-1.3b',
      messages: [
        { role: 'user', content: 'What is machine learning?' }
      ],
      max_tokens: 100,
      temperature: 0.7,
    }), {
      headers: { 'Content-Type': 'application/json' },
    });
    check(chatRes, {
      'chat completion status is 200': (r) => r.status === 200,
      'chat completion response time < 2000ms': (r) => r.timings.duration < 2000,
    });
  }
  // 20% embeddings
  else if (Math.random() < 0.286) {
    let embeddingsRes = http.post(`${API_URL}/v1/embeddings`, JSON.stringify({
      model: 'ryzanstein-embeddings',
      input: 'Sample text for embedding generation',
    }), {
      headers: { 'Content-Type': 'application/json' },
    });
    check(embeddingsRes, {
      'embeddings status is 200': (r) => r.status === 200,
      'embeddings response time < 500ms': (r) => r.timings.duration < 500,
    });
  }
  // 10% other endpoints
  else {
    let modelsRes = http.get(`${API_URL}/v1/models`);
    check(modelsRes, {
      'models status is 200': (r) => r.status === 200,
    });
  }

  sleep(Math.random() * 2);
}
