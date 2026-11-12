#pragma once

#include <agent-cpp/agent.hpp>

namespace agent::rag {

// Scales v to unit length (no-op for the zero vector).
void l2_normalize(std::span<float> v);

} // namespace agent::rag
