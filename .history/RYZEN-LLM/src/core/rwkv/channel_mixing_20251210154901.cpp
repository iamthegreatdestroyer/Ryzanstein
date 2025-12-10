#include "channel_mixing.h"

#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstdio>

namespace ryzen_llm::rwkv {

ChannelMixingLayer::ChannelMixingLayer(
  uint32_t hidden_dim,
  uint32_t layer_id,
  const ChannelMixingConfig& config
)
  : hidden_dim_(hidden_dim),
    layer_id_(layer_id),
    config_(config) {
  if (hidden_dim == 0) {
    throw std::invalid_argument("hidden_dim must be greater than 0");
  }
  if (config.hidden_dim != hidden_dim) {
    throw std::invalid_argument("config.hidden_dim must match hidden_dim parameter");
  }
  if (config.ff_expansion <= 0.0f || config.ff_expansion > 16.0f) {
    throw std::invalid_argument("ff_expansion must be in range (0, 16]");
  }
  if (config.value_gate < 0.0f || config.value_gate > 1.0f) {
    throw std::invalid_argument("value_gate must be in range [0, 1]");
  }
  if (config.key_gate < 0.0f || config.key_gate > 1.0f) {
    throw std::invalid_argument("key_gate must be in range [0, 1]");
  }
}

void ChannelMixingLayer::initialize() {
  printf("[CM] Starting initialize()\n");
  fflush(stdout);

  if (initialized_) {
    throw std::runtime_error("ChannelMixingLayer already initialized. Call reset_state() first.");
  }
  
  printf("[CM] Passed initialization check\n");
  fflush(stdout);

  // Check memory integrity
  printf("[CM] this pointer: %p\n", (void*)this);
  fflush(stdout);
  printf("[CM] hidden_dim_ address: %p\n", (void*)&hidden_dim_);
  fflush(stdout);
  printf("[CM] hidden_dim_ value: %u\n", hidden_dim_);
  fflush(stdout);
  printf("[CM] hidden_dim_ size_t: %zu\n", (size_t)hidden_dim_);
  fflush(stdout);

  // Check if value is reasonable
  if (hidden_dim_ == 0 || hidden_dim_ > 100000) {
    printf("[CM] ERROR: hidden_dim_ is unreasonable: %u\n", hidden_dim_);
    fflush(stdout);
    throw std::runtime_error("Invalid hidden_dim_");
  }

  // Simplest possible calculation - just use hardcoded value
  printf("[CM] Using hardcoded xavier scale = 0.125\n");
  fflush(stdout);
  float xavier_scale = 0.125f;
  printf("[CM] Xavier scale assigned: %f\n", xavier_scale);
  fflush(stdout);

  // Create RNG and distribution
  printf("[CM] Creating RNG with seed %u\n", layer_id_);
  fflush(stdout);
  std::mt19937 rng(layer_id_);
  
  printf("[CM] Creating distribution\n");
  fflush(stdout);
  std::uniform_real_distribution<float> dist(-xavier_scale, xavier_scale);
  
  printf("[CM] RNG ready, filling all buffers NOW\n");
  fflush(stdout);

  // Allocate shift parameter (learnable shift for temporal context)
  printf("[CM] Allocating shift (size=%u)\n", hidden_dim_);
  fflush(stdout);
  shift_.resize(hidden_dim_);
  printf("[CM] Filling shift\n");
  fflush(stdout);
  
  for (uint32_t i = 0; i < hidden_dim_; ++i) {
    shift_[i] = 0.01f * (i % 10 - 5) * 0.1f;
  }
  
  printf("[CM] Shift done\n");
  fflush(stdout);

  // Allocate key projection weights (hidden_dim x hidden_dim)
  printf("[CM] Allocating key_proj_w (size=%u)\n", hidden_dim_ * hidden_dim_);
  fflush(stdout);
  uint32_t proj_size = hidden_dim_ * hidden_dim_;
  key_proj_w_.resize(proj_size);
  printf("[CM] Filling key_proj_w\n");
  fflush(stdout);
  for (uint32_t i = 0; i < proj_size; ++i) {
    key_proj_w_[i] = dist(rng);
  }
  printf("[CM] key_proj_w done\n");
  fflush(stdout);

  // Allocate value projection weights (hidden_dim x hidden_dim)
  printf("[CM] Allocating value_proj_w (size=%u)\n", proj_size);
  fflush(stdout);
  value_proj_w_.resize(proj_size);
  printf("[CM] Filling value_proj_w\n");
  fflush(stdout);
  for (uint32_t i = 0; i < proj_size; ++i) {
    value_proj_w_[i] = dist(rng);
  }
  printf("[CM] value_proj_w done\n");
  fflush(stdout);

  // Allocate biases if enabled
  if (config_.use_bias) {
    printf("[CM] Allocating biases\n");
    fflush(stdout);
    key_proj_b_.resize(hidden_dim_);
    value_proj_b_.resize(hidden_dim_);
    for (uint32_t i = 0; i < hidden_dim_; ++i) {
      key_proj_b_[i] = dist(rng) * 0.1f;
      value_proj_b_[i] = dist(rng) * 0.1f;
    }
    printf("[CM] Biases done\n");
    fflush(stdout);
  }

  // Allocate internal buffers
  printf("[CM] Allocating internal buffers\n");
  fflush(stdout);
  prev_shifted_.resize(hidden_dim_, 0.0f);
  buffer_key_.resize(hidden_dim_, 0.0f);
  buffer_key_act_.resize(hidden_dim_, 0.0f);
  buffer_value_.resize(hidden_dim_, 0.0f);
  buffer_gate_.resize(hidden_dim_, 0.0f);
  printf("[CM] All buffers allocated\n");
  fflush(stdout);

  initialized_ = true;
  printf("[CM] initialize() complete\n");
  fflush(stdout);
}

void ChannelMixingLayer::shift_input_(
  const std::vector<float>& input,
  std::vector<float>& output
) {
  // Shift: blend current input with previous shifted state
  // output = (1 - shift_weight) * input + shift_weight * prev_shifted_
  for (uint32_t i = 0; i < hidden_dim_; ++i) {
    float shift_weight = 0.5f + (shift_[i] * 0.5f);  // Map shift to [0, 1]
    shift_weight = std::max(0.0f, std::min(1.0f, shift_weight));

    output[i] = (1.0f - shift_weight) * input[i] + shift_weight * prev_shifted_[i];
  }

  // Update previous state for next token
  std::copy(output.begin(), output.end(), prev_shifted_.begin());
}

void ChannelMixingLayer::apply_activation_(
  std::vector<float>& x,
  const std::string& activation_type
) {
  for (uint32_t i = 0; i < hidden_dim_; ++i) {
    x[i] = activation_fn_(x[i], activation_type);
  }
}

float ChannelMixingLayer::activation_fn_(float x, const std::string& activation_type) {
  if (activation_type == "relu") {
    return std::max(0.0f, x);
  } else if (activation_type == "gelu") {
    // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_pi = 0.7978845608f;  // sqrt(2/π)
    float cube_x = x * x * x;
    float tanh_input = sqrt_2_pi * (x + 0.044715f * cube_x);
    return x * 0.5f * (1.0f + std::tanh(tanh_input));
  } else if (activation_type == "swish") {
    // Swish: x * sigmoid(x) = x / (1 + exp(-x))
    return x / (1.0f + std::exp(-x));
  } else if (activation_type == "mish") {
    // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    float softplus = std::log(1.0f + std::exp(std::min(x, 20.0f)));  // Clamp to avoid overflow
    return x * std::tanh(softplus);
  }
  // Default to ReLU
  return std::max(0.0f, x);
}

void ChannelMixingLayer::matrix_multiply_(
  const std::vector<float>& A,
  const std::vector<float>& x,
  std::vector<float>& y,
  uint32_t m,
  uint32_t n,
  const std::vector<float>* bias
) {
  for (uint32_t i = 0; i < m; ++i) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < n; ++j) {
      sum += A[i * n + j] * x[j];
    }
    y[i] = sum;
    if (bias && !bias->empty()) {
      y[i] += (*bias)[i];
    }
  }
}

void ChannelMixingLayer::element_wise_multiply_(
  const std::vector<float>& a,
  const std::vector<float>& b,
  std::vector<float>& out
) {
  for (uint32_t i = 0; i < hidden_dim_; ++i) {
    out[i] = a[i] * b[i];
  }
}

void ChannelMixingLayer::element_wise_gate_(
  const std::vector<float>& value,
  const std::vector<float>& gate,
  std::vector<float>& out,
  float gate_strength
) {
  // Gate: out = value * (gate_strength + (1 - gate_strength) * gate)
  for (uint32_t i = 0; i < hidden_dim_; ++i) {
    float gate_val = gate_strength + (1.0f - gate_strength) * gate[i];
    gate_val = std::max(0.0f, std::min(1.0f, gate_val));
    out[i] = value[i] * gate_val;
  }
}

bool ChannelMixingLayer::forward(
  const std::vector<float>& input,
  std::vector<float>& output
) {
  if (!initialized_) {
    return false;
  }

  if (input.size() != hidden_dim_) {
    throw std::invalid_argument("Input size does not match hidden_dim");
  }

  if (output.size() != hidden_dim_) {
    throw std::invalid_argument("Output size does not match hidden_dim");
  }

  // Perform channel mixing
  if (!channel_mix(input, output)) {
    return false;
  }

  return true;
}

bool ChannelMixingLayer::channel_mix(
  const std::vector<float>& input,
  std::vector<float>& output
) {
  if (!initialized_) {
    return false;
  }

  // Step 1: Apply shift to blend with previous state
  std::vector<float> shifted(hidden_dim_);
  shift_input_(input, shifted);

  // Step 2: Key projection with activation (relu by default for channel mixing)
  std::vector<float> key_bias_ptr = config_.use_bias ? key_proj_b_ : std::vector<float>();
  matrix_multiply_(
    key_proj_w_, input, buffer_key_,
    hidden_dim_, hidden_dim_,
    config_.use_bias ? &key_proj_b_ : nullptr
  );

  // Step 3: Apply activation to key
  std::copy(buffer_key_.begin(), buffer_key_.end(), buffer_key_act_.begin());
  apply_activation_(buffer_key_act_, "relu");  // Key pathway uses ReLU

  // Step 4: Value projection
  matrix_multiply_(
    value_proj_w_, shifted, buffer_value_,
    hidden_dim_, hidden_dim_,
    config_.use_bias ? &value_proj_b_ : nullptr
  );

  // Step 5: Apply value gate (scales the value pathway)
  // Gate = sigmoid(key_proj) to create soft gating
  for (uint32_t i = 0; i < hidden_dim_; ++i) {
    // Sigmoid: 1 / (1 + exp(-x))
    float sig = 1.0f / (1.0f + std::exp(-buffer_key_act_[i]));
    buffer_gate_[i] = sig;
  }

  // Step 6: Gate the value with key signal
  element_wise_gate_(buffer_value_, buffer_gate_, output, config_.value_gate);

  return true;
}

bool ChannelMixingLayer::forward_sequence(
  const std::vector<float>& input_sequence,
  uint32_t seq_len,
  std::vector<float>& output_sequence
) {
  if (!initialized_) {
    return false;
  }

  uint32_t total_size = seq_len * hidden_dim_;
  if (input_sequence.size() != total_size) {
    throw std::invalid_argument("Input sequence size mismatch");
  }

  if (output_sequence.size() != total_size) {
    throw std::invalid_argument("Output sequence size mismatch");
  }

  std::vector<float> token_input(hidden_dim_);
  std::vector<float> token_output(hidden_dim_);

  // Process each token in sequence, threading state through
  for (uint32_t t = 0; t < seq_len; ++t) {
    uint32_t offset = t * hidden_dim_;

    // Extract token input
    std::copy(
      input_sequence.begin() + offset,
      input_sequence.begin() + offset + hidden_dim_,
      token_input.begin()
    );

    // Forward pass (includes state update)
    if (!forward(token_input, token_output)) {
      return false;
    }

    // Write token output
    std::copy(
      token_output.begin(),
      token_output.end(),
      output_sequence.begin() + offset
    );
  }

  return true;
}

void ChannelMixingLayer::reset_state() {
  std::fill(prev_shifted_.begin(), prev_shifted_.end(), 0.0f);
  std::fill(buffer_key_.begin(), buffer_key_.end(), 0.0f);
  std::fill(buffer_key_act_.begin(), buffer_key_act_.end(), 0.0f);
  std::fill(buffer_value_.begin(), buffer_value_.end(), 0.0f);
  std::fill(buffer_gate_.begin(), buffer_gate_.end(), 0.0f);
}

std::vector<uint8_t> ChannelMixingLayer::save_state() const {
  std::vector<uint8_t> state_data;

  // Write header (version + sizes)
  uint32_t version = 1;
  state_data.resize(sizeof(version));
  std::memcpy(state_data.data(), &version, sizeof(version));

  // Append prev_shifted_ state
  size_t prev_offset = state_data.size();
  state_data.resize(prev_offset + prev_shifted_.size() * sizeof(float));
  std::memcpy(
    state_data.data() + prev_offset,
    prev_shifted_.data(),
    prev_shifted_.size() * sizeof(float)
  );

  return state_data;
}

bool ChannelMixingLayer::load_state(const std::vector<uint8_t>& state) {
  if (state.size() < sizeof(uint32_t)) {
    return false;
  }

  uint32_t version = 0;
  std::memcpy(&version, state.data(), sizeof(version));

  if (version != 1) {
    return false;
  }

  size_t expected_size = sizeof(version) + prev_shifted_.size() * sizeof(float);
  if (state.size() != expected_size) {
    return false;
  }

  // Restore prev_shifted_ state
  std::memcpy(
    prev_shifted_.data(),
    state.data() + sizeof(version),
    prev_shifted_.size() * sizeof(float)
  );

  return true;
}

std::string ChannelMixingLayer::get_state_string() const {
  std::ostringstream oss;

  oss << "ChannelMixingLayer State:\n";
  oss << "  Hidden Dimension: " << hidden_dim_ << "\n";
  oss << "  Layer ID: " << layer_id_ << "\n";
  oss << "  Initialized: " << (initialized_ ? "true" : "false") << "\n";
  oss << "  Activation: " << config_.activation << "\n";
  oss << "  Value Gate: " << std::fixed << std::setprecision(4) << config_.value_gate << "\n";
  oss << "  Key Gate: " << std::fixed << std::setprecision(4) << config_.key_gate << "\n";
  oss << "  FF Expansion: " << std::fixed << std::setprecision(2) << config_.ff_expansion << "\n";
  oss << "  Use Bias: " << (config_.use_bias ? "true" : "false") << "\n";

  if (initialized_) {
    oss << "  Memory Allocated:\n";
    oss << "    shift_: " << shift_.size() * sizeof(float) / 1024.0f << " KB\n";
    oss << "    key_proj_w_: " << key_proj_w_.size() * sizeof(float) / 1024.0f << " KB\n";
    oss << "    value_proj_w_: " << value_proj_w_.size() * sizeof(float) / 1024.0f << " KB\n";

    uint32_t total_kb = 0;
    total_kb += static_cast<uint32_t>(shift_.size() * sizeof(float));
    total_kb += static_cast<uint32_t>(key_proj_w_.size() * sizeof(float));
    total_kb += static_cast<uint32_t>(value_proj_w_.size() * sizeof(float));
    if (!key_proj_b_.empty()) {
      total_kb += static_cast<uint32_t>(key_proj_b_.size() * sizeof(float));
      oss << "    key_proj_b_: " << key_proj_b_.size() * sizeof(float) / 1024.0f << " KB\n";
    }
    if (!value_proj_b_.empty()) {
      total_kb += static_cast<uint32_t>(value_proj_b_.size() * sizeof(float));
      oss << "    value_proj_b_: " << value_proj_b_.size() * sizeof(float) / 1024.0f << " KB\n";
    }
    oss << "    Total Weight Memory: " << total_kb / 1024.0f << " MB\n";

    // Sample prev_shifted values
    oss << "  State Samples (first 4 values of prev_shifted_):\n";
    for (uint32_t i = 0; i < std::min(4U, static_cast<uint32_t>(prev_shifted_.size())); ++i) {
      oss << "    [" << i << "]: " << std::fixed << std::setprecision(6) << prev_shifted_[i] << "\n";
    }
  }

  return oss.str();
}

}  // namespace ryzen_llm::rwkv
