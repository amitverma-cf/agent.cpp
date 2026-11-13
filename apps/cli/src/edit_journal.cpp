#include "edit_journal.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <simdjson.h>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

constexpr uintmax_t kMaxTracked = 1024 * 1024;

std::vector<std::string_view> split_lines(std::string_view s) {
    std::vector<std::string_view> lines;
    size_t at = 0;
    while (at < s.size()) {
        size_t end = s.find('\n', at);
        if (end == std::string_view::npos) end = s.size();
        lines.push_back(s.substr(at, end - at));
        at = end + 1;
    }
    return lines;
}

bool read_file(const std::string &path, std::string &content) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return false;
    if (fs::file_size(path, ec) > kMaxTracked) return false;
    std::ifstream in(path, std::ios::binary);
    content.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

struct Op {
    char type; // ' ', '-', '+'
    std::string_view text;
};

} // namespace

std::string unified_diff(std::string_view before, std::string_view after, std::string_view label) {
    if (before == after) return {};
    const auto a = split_lines(before);
    const auto b = split_lines(after);
    const std::string head = "--- a/" + std::string(label) + "\n+++ b/" + std::string(label) + "\n";
    if (static_cast<uint64_t>(a.size()) * b.size() > 4'000'000) return head + "(too large to show line by line)\n";

    // Longest common subsequence of the lines.
    const size_t n = a.size(), m = b.size();
    std::vector<uint32_t> lcs((n + 1) * (m + 1), 0);
    const auto at = [&](size_t i, size_t j) -> uint32_t & { return lcs[i * (m + 1) + j]; };
    for (size_t i = n; i-- > 0;)
        for (size_t j = m; j-- > 0;) at(i, j) = a[i] == b[j] ? at(i + 1, j + 1) + 1 : std::max(at(i + 1, j), at(i, j + 1));
    std::vector<Op> ops;
    size_t i = 0, j = 0;
    while (i < n || j < m) {
        if (i < n && j < m && a[i] == b[j]) ops.push_back({' ', a[i++]}), ++j;
        else if (j < m && (i == n || at(i, j + 1) >= at(i + 1, j))) ops.push_back({'+', b[j++]});
        else ops.push_back({'-', a[i++]});
    }

    // Lines of a and b consumed before each operation, for the hunk headers.
    std::vector<size_t> pa(ops.size() + 1, 0), pb(ops.size() + 1, 0);
    for (size_t k = 0; k < ops.size(); ++k) {
        pa[k + 1] = pa[k] + (ops[k].type != '+');
        pb[k + 1] = pb[k] + (ops[k].type != '-');
    }

    constexpr size_t kContext = 3;
    std::string out = head;
    size_t k = 0;
    while (k < ops.size()) {
        while (k < ops.size() && ops[k].type == ' ') ++k;
        if (k == ops.size()) break;
        const size_t start = k > kContext ? k - kContext : 0;
        size_t last = k; // the last changed operation of this hunk
        size_t scan = k;
        while (scan < ops.size()) {
            if (ops[scan].type != ' ') last = scan;
            else if (scan - last > 2 * kContext) break;
            ++scan;
        }
        const size_t end = std::min(ops.size(), last + kContext + 1);
        const size_t a_count = pa[end] - pa[start], b_count = pb[end] - pb[start];
        // A side with no lines names the line before the hunk, as diff does.
        out += "@@ -" + std::to_string(a_count == 0 ? pa[start] : pa[start] + 1) + "," + std::to_string(a_count) + " +" +
               std::to_string(b_count == 0 ? pb[start] : pb[start] + 1) + "," + std::to_string(b_count) + " @@\n";
        for (size_t x = start; x < end; ++x) {
            out += ops[x].type;
            out += ops[x].text;
            out += '\n';
        }
        k = end;
    }
    return out;
}

std::vector<std::string> files_changed_by_tool(std::string_view tool, std::string_view arguments_json) {
    std::vector<std::string> files;
    const bool one = tool == "write_file" || tool == "append_file" || tool == "delete_path";
    const bool two = tool == "move_path";
    if (!one && !two) return files;
    simdjson::dom::parser parser;
    simdjson::padded_string json(arguments_json);
    auto doc = parser.parse(json);
    if (doc.error()) return files;
    std::string_view value;
    for (const char *key : one ? std::vector<const char *>{"path"} : std::vector<const char *>{"from", "to"})
        if (doc[key].get(value) == simdjson::SUCCESS) files.emplace_back(value);
    return files;
}

std::string EditJournal::absolute(const std::string &path) const {
    fs::path p(path);
    if (p.is_relative()) p = fs::path(workspace_) / p;
    return p.lexically_normal().string();
}

std::string EditJournal::display(const std::string &path) const {
    std::error_code ec;
    const fs::path rel = fs::relative(path, workspace_, ec);
    const std::string text = ec || rel.empty() || *rel.begin() == ".." ? path : rel.generic_string();
    return text;
}

void EditJournal::before_tool(std::string_view tool, std::string_view arguments_json) {
    const std::vector<std::string> files = files_changed_by_tool(tool, arguments_json);
    if (files.empty()) return;
    ++group_;
    for (const std::string &file : files) {
        Entry e;
        e.path = absolute(file);
        e.group = group_;
        std::error_code ec;
        if (fs::is_directory(e.path, ec)) continue; // folders are not remembered
        e.existed = fs::exists(e.path, ec);
        if (e.existed && !read_file(e.path, e.before)) continue; // too large or unreadable
        entries_.push_back(std::move(e));
    }
}

std::string EditJournal::diff() const {
    std::string out;
    std::vector<std::string> seen;
    for (const Entry &e : entries_) {
        if (std::find(seen.begin(), seen.end(), e.path) != seen.end()) continue;
        seen.push_back(e.path);
        std::string now;
        const bool exists = read_file(e.path, now);
        if (!exists) now.clear();
        if (exists == e.existed && now == e.before) continue;
        if (!exists) out += "deleted: " + display(e.path) + "\n" + unified_diff(e.before, "", display(e.path));
        else if (!e.existed) out += "new file: " + display(e.path) + "\n" + unified_diff("", now, display(e.path));
        else out += unified_diff(e.before, now, display(e.path));
    }
    return out.empty() ? "no changes to files in this session" : out;
}

std::string EditJournal::undo() {
    if (entries_.empty()) return {};
    const int group = entries_.back().group;
    std::string report;
    while (!entries_.empty() && entries_.back().group == group) {
        const Entry e = std::move(entries_.back());
        entries_.pop_back();
        std::error_code ec;
        if (e.existed) {
            fs::create_directories(fs::path(e.path).parent_path(), ec);
            std::ofstream outf(e.path, std::ios::binary | std::ios::trunc);
            outf << e.before;
            report += "restored " + display(e.path) + "\n";
        } else {
            fs::remove(e.path, ec);
            report += "removed " + display(e.path) + " (it did not exist before)\n";
        }
    }
    return report;
}

} // namespace agent::cli
