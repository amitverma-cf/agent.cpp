#include "status_view.hpp"

#include <algorithm>
#include <chrono>

namespace agent::cli {

namespace {

int64_t wall_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// First line of s, control characters replaced by spaces, cut to max_len bytes (on a character boundary).
std::string first_line(std::string_view s, size_t max_len) {
    const size_t nl = s.find('\n');
    if (nl != std::string_view::npos) s = s.substr(0, nl);
    std::string out(s);
    for (char &c : out)
        if (static_cast<unsigned char>(c) < 0x20) c = ' ';
    if (out.size() > max_len) {
        size_t cut = max_len;
        while (cut > 0 && (static_cast<unsigned char>(out[cut]) & 0xC0) == 0x80) --cut;
        out.resize(cut);
        out += "...";
    }
    return out;
}

std::string fit_width(std::string line, int max_width) {
    if (max_width < 4 || line.size() <= static_cast<size_t>(max_width)) return line;
    size_t cut = static_cast<size_t>(max_width) - 3;
    while (cut > 0 && (static_cast<unsigned char>(line[cut]) & 0xC0) == 0x80) --cut;
    line.resize(cut);
    line += "...";
    return line;
}

std::string pad(std::string_view s, size_t width) {
    std::string out(s);
    if (out.size() < width) out.append(width - out.size(), ' ');
    return out;
}

int64_t elapsed_ms(const NodeStatus &row, int64_t now_ms) {
    const int64_t end = row.state == NodeState::Running ? now_ms : row.end_ms;
    return std::max<int64_t>(0, end - row.start_ms);
}

NodeStatus *find_row(std::vector<NodeStatus> &rows, std::string_view name) {
    for (auto &r : rows)
        if (r.name == name) return &r;
    return nullptr;
}

} // namespace

StatusMode choose_status_mode(bool json_flag, bool stdout_tty, bool stderr_tty, const char *term, const char *no_color) {
    if (json_flag) return StatusMode::Json;
    if (no_color && *no_color) return StatusMode::Plain;
    if (term && std::string_view(term) == "dumb") return StatusMode::Plain;
    return stdout_tty && stderr_tty ? StatusMode::Live : StatusMode::Plain;
}

const char *node_state_name(NodeState state) {
    switch (state) {
    case NodeState::Running: return "running";
    case NodeState::Done: return "done";
    case NodeState::Interrupted: return "interrupted";
    case NodeState::Error: return "error";
    }
    return "?";
}

const char *node_kind_name(GraphNodeKind kind) {
    switch (kind) {
    case GraphNodeKind::Agent: return "agent";
    case GraphNodeKind::Function: return "function";
    case GraphNodeKind::Tool: return "tool";
    case GraphNodeKind::Parallel: return "parallel";
    case GraphNodeKind::Loop: return "loop";
    }
    return "-";
}

std::string format_elapsed(int64_t ms) {
    if (ms < 0) ms = 0;
    char buf[32];
    if (ms < 60 * 1000) {
        std::snprintf(buf, sizeof(buf), "%lld.%llds", static_cast<long long>(ms / 1000), static_cast<long long>((ms % 1000) / 100));
    } else if (ms < 3600 * 1000) {
        const int64_t s = ms / 1000;
        std::snprintf(buf, sizeof(buf), "%lldm%02llds", static_cast<long long>(s / 60), static_cast<long long>(s % 60));
    } else {
        const int64_t m = ms / 60000;
        std::snprintf(buf, sizeof(buf), "%lldh%02lldm", static_cast<long long>(m / 60), static_cast<long long>(m % 60));
    }
    return buf;
}

const NodeStatus *apply_event(std::vector<NodeStatus> &rows, const GraphEvent &event, std::string_view kind) {
    switch (event.type) {
    case GraphEventType::NodeStart: {
        NodeStatus *row = find_row(rows, event.node);
        if (!row) {
            rows.push_back(NodeStatus{.name = event.node});
            row = &rows.back();
        }
        // A node that runs again (loop, resume) starts over.
        row->kind = std::string(kind.empty() ? "-" : kind);
        row->state = NodeState::Running;
        row->start_ms = event.timestamp_ms;
        row->end_ms = 0;
        row->last_tool.clear();
        row->detail.clear();
        return row;
    }
    case GraphEventType::ToolCall:
    case GraphEventType::ToolResult: {
        NodeStatus *row = find_row(rows, event.node);
        if (!row) return nullptr;
        row->last_tool = event.key;
        if (event.type == GraphEventType::ToolResult && event.payload.rfind("Error:", 0) == 0) row->last_tool += " (error)";
        return row;
    }
    case GraphEventType::NodeEnd:
    case GraphEventType::Interrupt:
    case GraphEventType::NodeFailed: {
        NodeStatus *row = find_row(rows, event.node);
        if (!row) return nullptr;
        row->state = event.type == GraphEventType::NodeEnd ? NodeState::Done
                     : event.type == GraphEventType::Interrupt ? NodeState::Interrupted
                                                               : NodeState::Error;
        row->end_ms = std::max(event.timestamp_ms, row->start_ms);
        row->detail = event.type == GraphEventType::NodeEnd ? std::string() : first_line(event.payload, 80);
        return row;
    }
    default: return nullptr;
    }
}

std::string render_status(std::span<const NodeStatus> rows, int64_t now_ms, int max_width, int max_rows) {
    std::vector<bool> hidden(rows.size(), false);
    size_t hidden_count = 0;
    if (max_rows > 0 && rows.size() > static_cast<size_t>(max_rows)) {
        const size_t want = rows.size() - static_cast<size_t>(max_rows) + 1; // one line is the summary
        for (size_t i = 0; i < rows.size() && hidden_count < want; ++i)
            if (rows[i].state == NodeState::Done) {
                hidden[i] = true;
                ++hidden_count;
            }
    }

    size_t name_w = 0, kind_w = 0, time_w = 0;
    std::vector<std::string> times(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        if (hidden[i]) continue;
        times[i] = format_elapsed(elapsed_ms(rows[i], now_ms));
        name_w = std::max(name_w, rows[i].name.size());
        kind_w = std::max(kind_w, rows[i].kind.size());
        time_w = std::max(time_w, times[i].size());
    }

    std::string out;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (hidden[i]) continue;
        const NodeStatus &r = rows[i];
        std::string line = pad(r.name, name_w) + "  " + pad(r.kind, kind_w) + "  " + pad(node_state_name(r.state), 11) + "  " +
                           std::string(time_w - times[i].size(), ' ') + times[i];
        if (!r.last_tool.empty()) line += "  tool: " + r.last_tool;
        if (!r.detail.empty()) line += "  - " + r.detail;
        out += fit_width(std::move(line), max_width) + "\n";
    }
    if (hidden_count) out += fit_width("(" + std::to_string(hidden_count) + " finished nodes hidden)", max_width) + "\n";
    return out;
}

std::string format_event_line(const GraphEvent &event, const NodeStatus &row) {
    const std::string head = "[node] " + row.name + " (" + row.kind + ") ";
    const std::string took = format_elapsed(elapsed_ms(row, row.end_ms));
    switch (event.type) {
    case GraphEventType::NodeStart: return head + "running";
    case GraphEventType::ToolCall: return head + "tool: " + event.key;
    case GraphEventType::NodeEnd: return head + "done in " + took;
    case GraphEventType::Interrupt: return head + "interrupted after " + took + ": " + row.detail;
    case GraphEventType::NodeFailed: return head + "error after " + took + ": " + row.detail;
    default: return {};
    }
}

std::string format_event_json(const GraphEvent &event, const NodeStatus &row) {
    const char *name = nullptr;
    switch (event.type) {
    case GraphEventType::NodeStart: name = "node_start"; break;
    case GraphEventType::ToolCall: name = "tool_call"; break;
    case GraphEventType::NodeEnd: name = "node_end"; break;
    case GraphEventType::Interrupt: name = "interrupt"; break;
    case GraphEventType::NodeFailed: name = "node_failed"; break;
    default: return {};
    }
    std::string out = std::string("{\"event\":\"") + name + "\",\"node\":\"" + json_escape(row.name) + "\",\"kind\":\"" +
                      json_escape(row.kind) + "\",\"ts\":" + std::to_string(event.timestamp_ms);
    if (event.type == GraphEventType::ToolCall) out += ",\"tool\":\"" + json_escape(event.key) + "\"";
    if (event.type != GraphEventType::NodeStart && event.type != GraphEventType::ToolCall)
        out += ",\"elapsed_ms\":" + std::to_string(elapsed_ms(row, row.end_ms));
    if (event.type == GraphEventType::Interrupt) out += ",\"reason\":\"" + json_escape(row.detail) + "\"";
    if (event.type == GraphEventType::NodeFailed) out += ",\"error\":\"" + json_escape(row.detail) + "\"";
    return out + "}";
}

StatusView::StatusView(std::FILE *out, StatusMode mode, const Graph &graph, int width) : out_(out), mode_(mode), width_(width) {
    for (const GraphNode &n : graph.nodes) kinds_.emplace_back(n.name, node_kind_name(n.kind));
    if (mode_ == StatusMode::Live) ticker_ = std::thread([this] { tick_loop(); });
}

StatusView::~StatusView() { finish(); }

void StatusView::on_graph_event(const GraphEvent &event, void *user_data) { static_cast<StatusView *>(user_data)->on_event(event); }

void StatusView::on_event(const GraphEvent &event) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string_view kind;
    for (const auto &[name, k] : kinds_)
        if (name == event.node) kind = k;
    const NodeStatus *row = apply_event(rows_, event, kind);
    if (!row) return;
    if (mode_ == StatusMode::Live) {
        draw_locked();
        return;
    }
    std::string line = mode_ == StatusMode::Json ? format_event_json(event, *row) : format_event_line(event, *row);
    if (line.empty()) return;
    std::fprintf(out_, "%s\n", line.c_str());
    std::fflush(out_);
}

// Moves up over the table drawn last time and rewrites it line by line. Rows are never removed, so
// the new table is never shorter than the old one.
void StatusView::draw_locked() {
    if (held_) return;
    const std::string text = render_status(rows_, wall_now_ms(), width_, 20);
    std::string frame;
    if (drawn_lines_ > 0) frame += "\x1b[" + std::to_string(drawn_lines_) + "A";
    int lines = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t nl = text.find('\n', pos);
        frame += "\r\x1b[2K" + text.substr(pos, nl - pos) + "\n";
        pos = nl + 1;
        ++lines;
    }
    drawn_lines_ = lines;
    std::fwrite(frame.data(), 1, frame.size(), out_);
    std::fflush(out_);
}

void StatusView::erase_locked() {
    if (drawn_lines_ > 0) {
        std::fprintf(out_, "\x1b[%dA\r\x1b[J", drawn_lines_);
        std::fflush(out_);
    }
    drawn_lines_ = 0;
}

void StatusView::tick_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
        wake_.wait_for(lock, std::chrono::milliseconds(250));
        if (stop_) break;
        const bool running = std::any_of(rows_.begin(), rows_.end(), [](const NodeStatus &r) { return r.state == NodeState::Running; });
        if (running) draw_locked();
    }
}

void StatusView::finish() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) return;
        stop_ = true;
        held_ = false;
    }
    wake_.notify_all();
    if (ticker_.joinable()) ticker_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ == StatusMode::Live) draw_locked();
}

void StatusView::hold() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != StatusMode::Live || held_) return;
    erase_locked();
    held_ = true;
}

void StatusView::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != StatusMode::Live || !held_) return;
    held_ = false;
    draw_locked();
}

std::vector<NodeStatus> StatusView::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return rows_;
}

} // namespace agent::cli
