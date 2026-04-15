import http from 'k6/http';
import { check, sleep } from 'k6';

// Smoke Test: Quick sanity check of all endpoints
// Tests baseline functionality with 1 VU for 30 seconds

export const options = {
  vus: 1,
  duration: '30s',
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.1'],
  },
};

const API_URL = __ENV.API_URL || 'http://localhost:8000';

export default function () {
  // Test 1: Health endpoint
  let healthRes = http.get(`${API_URL}/health`);
  check(healthRes, {
    'health status is 200': (r) => r.status === 200,
    'health response time < 100ms': (r) => r.timings.duration < 100,
  });

  sleep(1);

  // Test 2: List models endpoint
  let modelsRes = http.get(`${API_URL}/v1/models`);
  check(modelsRes, {
    'models status is 200': (r) => r.status === 200,
    'models response time < 200ms': (r) => r.timings.duration < 200,
  });

  sleep(1);

  // Test 3: Chat completion endpoint
  let chatRes = http.post(`${API_URL}/v1/chat/completions`, JSON.stringify({
    model: 'ryzanstein-1.3b',
    messages: [
      { role: 'user', content: 'Hello, how are you?' }
    ],
    max_tokens: 50,
    temperature: 0.7,
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  check(chatRes, {
    'chat completion status is 200': (r) => r.status === 200,
    'chat completion response time < 2000ms': (r) => r.timings.duration < 2000,
  });

  sleep(1);

  // Test 4: Embeddings endpoint
  let embeddingsRes = http.post(`${API_URL}/v1/embeddings`, JSON.stringify({
    model: 'ryzanstein-embeddings',
    input: 'The quick brown fox jumps over the lazy dog',
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  check(embeddingsRes, {
    'embeddings status is 200': (r) => r.status === 200,
    'embeddings response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);

  // Test 5: Root endpoint
  let rootRes = http.get(`${API_URL}/`);
  check(rootRes, {
    'root status is 200': (r) => r.status === 200,
    'root response time < 100ms': (r) => r.timings.duration < 100,
  });

  sleep(2);
}
