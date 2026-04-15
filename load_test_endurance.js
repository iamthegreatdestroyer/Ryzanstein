/**
 * Ryzanstein LLM API - Endurance Test
 *
 * Purpose: Detect memory leaks, resource exhaustion, or performance degradation over time
 * Load Pattern: 25 VUs sustained for 1 hour
 * Duration: 60 minutes
 * Expected Results: Stable performance, no degradation, no resource leaks
 */

import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 25,
  duration: '60m',
  thresholds: {
    http_req_duration: ['p(95)<1000', 'p(99)<2000'],
    http_req_failed: ['rate<0.02'],
  },
};

const API_URL = __ENV.API_URL || 'http://localhost:8000';

export default function () {
  // Primary workload: chat completions
  let chatRes = http.post(`${API_URL}/v1/chat/completions`, JSON.stringify({
    model: 'ryzanstein-1.3b',
    messages: [
      { role: 'user', content: `Endurance test iteration ${__ITER}` }
    ],
    max_tokens: 100,
    temperature: 0.7,
  }), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '5s',
  });

  check(chatRes, {
    'status is 200': (r) => r.status === 200,
    'response has content': (r) => r.body.length > 0,
    'latency < 1000ms': (r) => r.timings.duration < 1000,
  });

  sleep(2);
}
