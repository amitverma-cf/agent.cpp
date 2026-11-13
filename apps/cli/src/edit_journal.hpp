#pragma once

// /diff and /undo in the chat: before a built-in tool changes a file, the file's content is remembered, so the session's edits can be
// shown as a unified diff and put back one tool call at a time. Only files up to 1 MiB are remembered; folders are not.

#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// Unified diff of two texts with 3 lines of context; empty when they are equal.
std::string unified_diff(std::string_view before, std::string_view after, std::string_view label);

// The files a built-in tool call changes: write_file / append_file / delete_path -> path, move_path -> from and to.
std::vector<std::string> files_changed_by_tool(std::string_view tool, std::string_view arguments_json);

class EditJournal {
  public:
    explicit EditJournal(std::string workspace) : workspace_(std::move(workspace)) {}

    // Called before a tool runs: remembers the files it is about to change.
    void before_tool(std::string_view tool, std::string_view arguments_json);
    // The edits of this session as a unified diff against the files as they are now; "no changes" when there are none.
    std::string diff() const;
    // Puts the files of the last tool call back (a file the call created is deleted). Empty when there is nothing to undo.
    std::string undo();
    size_t size() const { return entries_.size(); }

  private:
    struct Entry {
        std::string path; // absolute
        bool existed = false;
        std::string before; // the content before the call
        int group = 0;      // the tool call
    };
    std::string absolute(const std::string &path) const;
    std::string display(const std::string &path) const;

    std::string workspace_;
    std::vector<Entry> entries_;
    int group_ = 0;
};

} // namespace agent::cli
