#pragma once

// Live status of a running graph for `agent run`: one line per node (name, kind, state, elapsed time,
// last tool call). Fed by GraphRun::on_event, which parallel branches call from worker threads.
// render_status and the line formatters are pure functions so they can be tested without a terminal;
// StatusView adds the table, the mutex and the output.

#include "cli_core.hpp"

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace agent::cli {

enum class NodeState { Running, Done, Interrupted, Error };

struct NodeStatus {
    std::string name;
    std::string kind;      // "agent", "function", "tool", "parallel", "loop" ("-" when the node is unknown)
    NodeState state = NodeState::Running;
    int64_t start_ms = 0;  // wall clock, ms since the Unix epoch (GraphEvent::timestamp_ms)
    int64_t end_ms = 0;    // when it stopped; 0 while running
    std::string last_tool; // name of the latest tool call, "name (error)" if it returned an error
    std::string detail;    // interrupt reason or error message
};

enum class StatusMode {
    Live,  // redraw the table in place with ANSI escapes
    Plain, // one text line per event
    Json,  // one JSON object per event
};

// Live only when the output is an interactive terminal: stdout and stderr are TTYs, TERM is not
// "dumb" (an unset TERM is accepted, as on Windows) and NO_COLOR is unset or empty. json_flag (--json)
// wins over everything. term / no_color may be null.
StatusMode choose_status_mode(bool json_flag, bool stdout_tty, bool stderr_tty, const char *term, const char *no_color);

const char *node_state_name(NodeState state); // "running", "done", "interrupted", "error"
const char *node_kind_name(GraphNodeKind kind);
// "0.4s", "12.3s", "2m05s", "1h02m".
std::string format_elapsed(int64_t ms);

// Applies one graph event to the table. Other event types are ignored. `kind` is used when the event
// creates a row. Returns the row the event changed, or nullptr.
const NodeStatus *apply_event(std::vector<NodeStatus> &rows, const GraphEvent &event, std::string_view kind);

// The table as text, one line per row, each ending in '\n':
//   name   kind      state        elapsed  tool: <last tool>  - <detail>
// Running rows count up to now_ms. A line longer than max_width columns (0 = no limit) is cut with
// "...". When there are more than max_rows rows (0 = no limit), the oldest finished rows are left out
// and a last line says how many.
std::string render_status(std::span<const NodeStatus> rows, int64_t now_ms, int max_width = 0, int max_rows = 0);

// One-line text / JSON form of an event, without trailing newline; empty for events that are not shown
// (state writes, checkpoints, tool results). row is the table row apply_event returned for the event.
std::string format_event_line(const GraphEvent &event, const NodeStatus &row);
std::string format_event_json(const GraphEvent &event, const NodeStatus &row);

class StatusView {
  public:
    // width: terminal columns for Live (0 = unlimited). Nodes are looked up in graph for their kind.
    StatusView(std::FILE *out, StatusMode mode, const Graph &graph, int width = 0);
    ~StatusView();
    StatusView(const StatusView &) = delete;
    StatusView &operator=(const StatusView &) = delete;

    // Thread-safe.
    void on_event(const GraphEvent &event);
    // GraphRun::on_event adapter; user_data is the StatusView.
    static void on_graph_event(const GraphEvent &event, void *user_data);

    // Live: stops the clock thread and draws the table one last time. Safe to call twice.
    void finish();
    // Live: erases the table and stops drawing, so a prompt can use the terminal; release() draws it
    // again below whatever was printed meanwhile.
    void hold();
    void release();

    std::vector<NodeStatus> snapshot() const;

  private:
    void draw_locked();
    void erase_locked();
    void tick_loop();

    std::FILE *out_;
    StatusMode mode_;
    int width_;
    std::vector<std::pair<std::string, std::string>> kinds_; // node name -> kind name

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::vector<NodeStatus> rows_;
    int drawn_lines_ = 0;
    bool held_ = false;
    bool stop_ = false;
    std::thread ticker_;
};

} // namespace agent::cli
