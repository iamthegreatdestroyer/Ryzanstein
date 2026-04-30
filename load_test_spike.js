/**
 * Ryzanstein LLM API - Spike Test
 *
 * Purpose: Test system's ability to handle sudden load spikes
 * Load Pattern: 10 → 100 → 10 VUs (sudden spike and recovery)
 * Duration: 5 minutes
 * Expected Results: System handles spike gracefully, recovers quickly
 */

import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 10 },   // Normal load
    { duration: '1m', target: 100 },   // Sudden spike (realistic)
    { duration: '3m', target: 10 },    // Return to normal
    { duration: '30s', target: 0 },    // Cool down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000', 'p(99)<5000'],
    http_req_failed: ['rate<0.1'],
  },
};

const API_URL = __ENV.API_URL || 'http://localhost:8000';

export default function () {
  // Spike test focuses on chat completions
  let chatRes = http.post(`${API_URL}/v1/chat/completions`, JSON.stringify({
    model: 'ryzanstein-1.3b',
    messages: [
      { role: 'user', content: 'Spike test request' }
    ],
    max_tokens: 64,
    temperature: 0.7,
  }), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '10s',
  });

  check(chatRes, {
    'status is 200': (r) => r.status === 200,
    'status is not 500': (r) => r.status !== 500,
    'response time acceptable': (r) => r.timings.duration < 5000,
  });

  sleep(0.5);
}
