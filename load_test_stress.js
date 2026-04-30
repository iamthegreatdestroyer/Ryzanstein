/**
 * Ryzanstein LLM API - Stress Test
 *
 * Purpose: Push API to identify performance limits and breaking point
 * Load Pattern: 100 → 500 → 1000 VUs (aggressive ramp-up over 30 minutes)
 * Duration: 30 minutes
 * Expected Results: Identify breaking point, measure recovery behavior
 */

import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 VUs
    { duration: '5m', target: 500 },   // Ramp to 500 VUs
    { duration: '10m', target: 1000 }, // Ramp to 1000 VUs (identify limits)
    { duration: '10m', target: 100 },  // Ramp down for recovery
    { duration: '3m', target: 0 },     // Cool down
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000', 'p(99)<10000'],  // More realistic for 1000 VU stress test
    http_req_failed: ['rate<0.25'],  // Allow up to 25% failure under extreme stress
  },
};

const API_URL = __ENV.API_URL || 'http://localhost:8000';

export default function () {
  // Mix of endpoints simulating realistic production workload
  const requestType = Math.random();

  if (requestType < 0.7) {
    // 70% chat completions (main workload)
    let chatRes = http.post(`${API_URL}/v1/chat/completions`, JSON.stringify({
      model: 'ryzanstein-1.3b',
      messages: [
        { role: 'user', content: 'Stress test query' }
      ],
      max_tokens: 64,
      temperature: 0.5,
    }), {
      headers: { 'Content-Type': 'application/json' },
      timeout: '10s',
    });

    check(chatRes, {
      'chat status is success': (r) => r.status === 200,
      'chat response valid': (r) => r.body.length > 0,
    });
  } else if (requestType < 0.85) {
    // 15% embeddings
    let embeddingsRes = http.post(`${API_URL}/v1/embeddings`, JSON.stringify({
      model: 'ryzanstein-embeddings',
      input: 'Stress test embedding',
    }), {
      headers: { 'Content-Type': 'application/json' },
      timeout: '10s',
    });

    check(embeddingsRes, {
      'embeddings status is success': (r) => r.status === 200,
    });
  } else {
    // 15% health and model checks
    let healthRes = http.get(`${API_URL}/health`, { timeout: '10s' });
    check(healthRes, {
      'health status is success': (r) => r.status === 200,
    });
  }

  sleep(0.1);
}
