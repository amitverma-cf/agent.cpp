#pragma once

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace agent {

// JSON (de)serialisation of a Conversation history, shared by save/load_conversation and the graph
// checkpointer. The format is a JSON array of {role, content, tool_call_id, tool_calls, media?}.
void encode_history_json(const Conversation &state, std::string &json);
Result<std::vector<Message>> decode_history_json(std::string_view json);

} // namespace agent
