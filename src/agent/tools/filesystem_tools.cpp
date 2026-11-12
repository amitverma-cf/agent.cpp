#include "../utils/utils.hpp"
#include "hal/path.hpp"
#include "path_gate.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <simdjson.h>
#include <system_error>
#include <utility>
#include <vector>

namespace agent::tools {

namespace {

bool parse_obj(std::string_view args, simdjson::ondemand::parser &parser, simdjson::padded_string &json, simdjson::ondemand::document &doc,
               simdjson::ondemand::object &obj) {
    json = simdjson::padded_string(args.data(), args.size());
    if (parser.iterate(json).get(doc)) return false;
    if (doc.get_object().get(obj)) return false;
    return true;
}

Result<std::filesystem::path> resolve_path(const WorkspaceContext &ctx, std::string_view rel, bool write = false) {
    if (!ctx.workspace_dir) return fail<std::filesystem::path>(ErrorCode::InvalidConfig, "workspace_dir is null.");
    auto resolved = hal::resolve_under_workspace(*ctx.workspace_dir, rel, ctx.sandbox);
    if (resolved.ok || resolved.error.code != ErrorCode::SandboxViolation || !ctx.sandbox) return resolved;
    // Outside the workspace: allowed only by a grant or by the host's gate.
    std::filesystem::path wanted{std::string(rel)};
    if (!wanted.is_absolute()) wanted = std::filesystem::path(*ctx.workspace_dir) / wanted;
    std::error_code ec;
    wanted = std::filesystem::weakly_canonical(wanted, ec);
    if (!ec && path_access_allowed(ctx, "files", wanted, write)) return ok(std::move(wanted));
    resolved.error.message += " Grant access with `agent sandbox grant` or approve it when asked.";
    return resolved;
}

} // namespace

namespace {

constexpr int64_t kReadDefaultBytes = 16384;
constexpr int64_t kReadMaxBytes = 4 * 1024 * 1024; // a larger max_bytes is clamped

// Media types read_file hands back as attachments instead of text, by file extension.
std::string_view media_mime_for(const std::filesystem::path &path) {
    std::string ext = path.extension().string();
    for (char &c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    static constexpr std::pair<std::string_view, std::string_view> kTypes[] = {
        {".png", "image/png"},        {".jpg", "image/jpeg"},     {".jpeg", "image/jpeg"}, {".gif", "image/gif"},
        {".webp", "image/webp"},      {".bmp", "image/bmp"},      {".tga", "image/x-tga"}, {".wav", "audio/wav"},
        {".mp3", "audio/mpeg"},       {".flac", "audio/flac"},    {".ogg", "audio/ogg"},   {".opus", "audio/opus"},
        {".m4a", "audio/mp4"},        {".mp4", "video/mp4"},      {".webm", "video/webm"}, {".mov", "video/quicktime"},
        {".mkv", "video/x-matroska"}, {".avi", "video/x-msvideo"}};
    for (const auto &[e, mime] : kTypes)
        if (ext == e) return mime;
    return {};
}

// A media file outside the workspace is copied into <workspace>/artifacts/ first: a sandboxed
// out-of-process engine can only read the workspace (and the files the history keeps pointing at
// should not change under it).
Result<std::string> media_path_for_engine(const WorkspaceContext &ctx, const std::filesystem::path &full) {
    std::error_code ec;
    auto abs = std::filesystem::weakly_canonical(full, ec);
    if (ec) abs = std::filesystem::absolute(full, ec);
    if (hal::resolve_under_workspace(*ctx.workspace_dir, abs.string(), true).ok) return ok(abs.string());

    auto dir = std::filesystem::absolute(std::filesystem::path(*ctx.workspace_dir) / "artifacts", ec);
    std::filesystem::create_directories(dir, ec);
    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto dest = dir / ("read-" + std::to_string(stamp) + "-" + full.filename().string());
    std::filesystem::copy_file(full, dest, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) return fail<std::string>(ErrorCode::FilesystemError, "Cannot copy " + full.string() + " into the workspace: " + ec.message());
    return ok(dest.string());
}

} // namespace

// Text files come back as text (invalid UTF-8 replaced, cut at a character boundary); images, audio
// and video come back as media, so a model with that input modality sees / hears the file.
static Result<ToolResult> read_file_cb(std::string_view args, void *ud) {
    const auto &ctx = *static_cast<const WorkspaceContext *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj)) return fail<ToolResult>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv;
    int64_t max_bytes_i = kReadDefaultBytes;
    int64_t offset_i = 0;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error()) continue;
        if (k.value() == "path") {
            if (field.value().get_string().get(path_sv)) path_sv = {};
        } else if (k.value() == "max_bytes") {
            int64_t v;
            if (!field.value().get_int64().get(v) && v >= 0) max_bytes_i = std::min(v, kReadMaxBytes);
        } else if (k.value() == "offset") {
            int64_t v;
            if (!field.value().get_int64().get(v) && v >= 0) offset_i = v;
        }
    }
    if (path_sv.empty()) return fail<ToolResult>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full_res = resolve_path(ctx, path_sv);
    if (!full_res.ok) return fail<ToolResult>(full_res.error.code, full_res.error.message);
    auto full = std::move(full_res.value);

    std::error_code ec;
    if (!std::filesystem::exists(full, ec)) return fail<ToolResult>(ErrorCode::KeyNotFound, "Not found: " + full.string());
    if (std::filesystem::is_directory(full, ec))
        return fail<ToolResult>(ErrorCode::InvalidConfig, "Is a directory (use list_dir): " + full.string());
    const uintmax_t file_size = std::filesystem::file_size(full, ec);
    const uint64_t total = ec ? 0 : static_cast<uint64_t>(file_size);

    ToolResult out;
    if (std::string_view mime = media_mime_for(full); !mime.empty()) {
        auto path = media_path_for_engine(ctx, full);
        if (!path.ok) return fail<ToolResult>(path.error.code, path.error.message);
        out.text = std::string(mime.substr(0, mime.find('/'))) + " file " + full.filename().string() + " (" + std::to_string(total) +
                   " bytes), attached";
        out.media.push_back(Media{.path = std::move(path.value), .mime = std::string(mime)});
        return ok(std::move(out));
    }

    std::ifstream f(full, std::ios::binary);
    if (!f) return fail<ToolResult>(ErrorCode::FilesystemError, "Cannot open: " + full.string());
    const auto offset = static_cast<uint64_t>(offset_i);
    if (offset > 0) f.seekg(static_cast<std::streamoff>(offset));

    // One byte of look-ahead tells "exactly max_bytes left" from "more to come".
    const auto max_bytes = static_cast<size_t>(max_bytes_i);
    std::string content(max_bytes + 1, '\0');
    if (offset < total || total == 0) f.read(content.data(), static_cast<std::streamsize>(content.size()));
    content.resize(static_cast<size_t>(std::max<std::streamsize>(f.gcount(), 0)));
    const bool more = content.size() > max_bytes;
    if (more) content.resize(utils::utf8_prefix_length(content, max_bytes));

    if (content.find('\0') != std::string::npos) {
        out.text = "[binary file " + full.filename().string() + ", " + std::to_string(total) + " bytes; not shown as text]";
        return ok(std::move(out));
    }
    const uint64_t next = offset + content.size(); // bytes consumed, before invalid bytes are replaced
    utils::sanitize_utf8(content);
    out.text = std::move(content);
    if (more) {
        out.text += "\n[truncated at " + std::to_string(max_bytes) + " bytes]";
        out.text += "\n[" + std::to_string(total) + " bytes in total; read more with \"offset\": " + std::to_string(next) + "]";
    }
    return ok(std::move(out));
}
static Result<std::string> write_file_cb(std::string_view args, void *ud) {
    const auto &ctx = *static_cast<const WorkspaceContext *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj)) return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv, content_sv;
    bool create_dirs = true;
    bool has_content = false;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error()) continue;
        auto kv = k.value();
        if (kv == "path") {
            if (field.value().get_string().get(path_sv)) path_sv = {};
        } else if (kv == "content") {
            has_content = !field.value().get_string().get(content_sv);
        } else if (kv == "create_dirs") {
            bool b;
            if (!field.value().get_bool().get(b)) create_dirs = b;
        }
    }
    if (path_sv.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");
    // A model that forgets "content" (or sends a non-string) must not silently truncate the file.
    if (!has_content) return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'content' (a string)");

    auto full_res = resolve_path(ctx, path_sv, true);
    if (!full_res.ok) return fail<std::string>(full_res.error.code, full_res.error.message);
    auto full = std::move(full_res.value);

    if (create_dirs) {
        std::error_code ec;
        std::filesystem::create_directories(full.parent_path(), ec);
    }

    std::ofstream f(full, std::ios::binary | std::ios::trunc);
    if (!f) return fail<std::string>(ErrorCode::FilesystemError, "Cannot write: " + full.string());
    f.write(content_sv.data(), static_cast<std::streamsize>(content_sv.size()));
    return ok("wrote " + std::to_string(content_sv.size()) + " bytes to " + full.string());
}

static Result<std::string> append_file_cb(std::string_view args, void *ud) {
    const auto &ctx = *static_cast<const WorkspaceContext *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj)) return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv, content_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error()) continue;
        if (k.value() == "path") {
            if (field.value().get_string().get(path_sv)) path_sv = {};
        } else if (k.value() == "content") {
            if (field.value().get_string().get(content_sv)) content_sv = {};
        }
    }
    if (path_sv.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full_res = resolve_path(ctx, path_sv, true);
    if (!full_res.ok) return fail<std::string>(full_res.error.code, full_res.error.message);
    auto full = std::move(full_res.value);

    std::error_code ec;
    std::filesystem::create_directories(full.parent_path(), ec);

    std::ofstream f(full, std::ios::binary | std::ios::app);
    if (!f) return fail<std::string>(ErrorCode::FilesystemError, "Cannot open: " + full.string());
    f.write(content_sv.data(), static_cast<std::streamsize>(content_sv.size()));
    return ok("appended " + std::to_string(content_sv.size()) + " bytes to " + full.string());
}

static Result<std::string> list_dir_cb(std::string_view args, void *ud) {
    const auto &ctx = *static_cast<const WorkspaceContext *>(ud);
    std::string path_str;

    if (!args.empty()) {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json;
        simdjson::ondemand::document doc;
        simdjson::ondemand::object obj;
        if (parse_obj(args, parser, json, doc, obj)) {
            for (auto field : obj) {
                auto k = field.unescaped_key();
                if (!k.error() && k.value() == "path") {
                    std::string_view sv;
                    if (!field.value().get_string().get(sv)) path_str = std::string(sv);
                }
            }
        }
    }

    auto full_res = resolve_path(ctx, path_str.empty() ? "." : path_str);
    if (!full_res.ok) return fail<std::string>(full_res.error.code, full_res.error.message);
    auto dir = std::move(full_res.value);

    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return fail<std::string>(ErrorCode::KeyNotFound, "Not a directory: " + dir.string());

    struct Entry {
        std::string name;
        bool is_dir = false;
        bool has_size = false;
        uintmax_t size = 0;
    };
    constexpr size_t kMaxEntries = 1000;
    std::vector<Entry> entries;
    bool truncated = false;
    // Range-based for would use directory_iterator::operator++() (the throwing overload) on
    // every iteration, even though construction here is non-throwing -- advance manually with
    // the error_code overload so a mid-scan filesystem error (e.g. a permission-denied entry)
    // can't throw an uncaught std::filesystem::filesystem_error.
    for (auto it = std::filesystem::directory_iterator(dir, ec); !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        if (entries.size() == kMaxEntries) {
            truncated = true;
            break;
        }
        const auto &entry = *it;
        std::error_code entry_ec;
        Entry e;
        e.name = entry.path().filename().string();
        e.is_dir = entry.is_directory(entry_ec);
        if (!e.is_dir) {
            e.size = entry.file_size(entry_ec);
            e.has_size = !entry_ec;
        }
        entries.push_back(std::move(e));
    }
    // Directory order is filesystem-dependent; sorted output is stable across runs (and replays).
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.name < b.name; });

    std::string result = "[";
    for (size_t i = 0; i < entries.size(); ++i) {
        const Entry &e = entries[i];
        if (i) result += ',';
        result += "{\"name\":\"";
        std::string name = e.name;
        utils::sanitize_utf8(name);
        utils::escape_json_string(name, result);
        result += "\",\"type\":\"";
        result += e.is_dir ? "dir" : "file";
        result += '"';
        if (e.has_size) result += ",\"size\":" + std::to_string(e.size);
        result += '}';
    }
    if (truncated) result += ",{\"truncated\":true,\"shown\":" + std::to_string(kMaxEntries) + "}";
    result += ']';
    return ok(std::move(result));
}

static Result<std::string> create_dir_cb(std::string_view args, void *ud) {
    const auto &ctx = *static_cast<const WorkspaceContext *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj)) return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (!k.error() && k.value() == "path") {
            if (field.value().get_string().get(path_sv)) path_sv = {};
        }
    }
    if (path_sv.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full_res = resolve_path(ctx, path_sv, true);
    if (!full_res.ok) return fail<std::string>(full_res.error.code, full_res.error.message);
    auto full = std::move(full_res.value);

    std::error_code ec;
    std::filesystem::create_directories(full, ec);
    if (ec) return fail<std::string>(ErrorCode::FilesystemError, "Failed: " + ec.message());
    return ok("created " + full.string());
}

static Result<std::string> delete_path_cb(std::string_view args, void *ud) {
    const auto &ctx = *static_cast<const WorkspaceContext *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj)) return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (!k.error() && k.value() == "path") {
            if (field.value().get_string().get(path_sv)) path_sv = {};
        }
    }
    if (path_sv.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full_res = resolve_path(ctx, path_sv, true);
    if (!full_res.ok) return fail<std::string>(full_res.error.code, full_res.error.message);
    auto full = std::move(full_res.value);

    if (ctx.workspace_dir && hal::is_workspace_root(*ctx.workspace_dir, full))
        return fail<std::string>(ErrorCode::SandboxViolation, "Cannot delete the workspace root directory.");

    std::error_code ec;
    auto count = std::filesystem::remove_all(full, ec);
    if (ec) return fail<std::string>(ErrorCode::FilesystemError, "Failed: " + ec.message());
    return ok("deleted " + std::to_string(count) + " entries at " + full.string());
}

static Result<std::string> move_path_cb(std::string_view args, void *ud) {
    const auto &ctx = *static_cast<const WorkspaceContext *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj)) return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view from_sv, to_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error()) continue;
        if (k.value() == "from") {
            if (field.value().get_string().get(from_sv)) from_sv = {};
        } else if (k.value() == "to") {
            if (field.value().get_string().get(to_sv)) to_sv = {};
        }
    }
    if (from_sv.empty() || to_sv.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'from' or 'to'");

    auto src_res = resolve_path(ctx, from_sv, true);
    if (!src_res.ok) return fail<std::string>(src_res.error.code, src_res.error.message);
    auto dst_res = resolve_path(ctx, to_sv, true);
    if (!dst_res.ok) return fail<std::string>(dst_res.error.code, dst_res.error.message);
    if (ctx.workspace_dir && hal::is_workspace_root(*ctx.workspace_dir, src_res.value))
        return fail<std::string>(ErrorCode::SandboxViolation, "Cannot move the workspace root directory.");

    std::error_code ec;
    std::filesystem::create_directories(dst_res.value.parent_path(), ec);
    std::filesystem::rename(src_res.value, dst_res.value, ec);
    if (ec) return fail<std::string>(ErrorCode::FilesystemError, "Failed: " + ec.message());
    return ok("moved " + src_res.value.string() + " -> " + dst_res.value.string());
}

static Result<std::string> file_info_cb(std::string_view args, void *ud) {
    const auto &ctx = *static_cast<const WorkspaceContext *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj)) return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (!k.error() && k.value() == "path") {
            if (field.value().get_string().get(path_sv)) path_sv = {};
        }
    }
    if (path_sv.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full_res = resolve_path(ctx, path_sv);
    if (!full_res.ok) return fail<std::string>(full_res.error.code, full_res.error.message);
    auto full = std::move(full_res.value);

    std::error_code ec;
    auto st = std::filesystem::status(full, ec);
    if (ec || st.type() == std::filesystem::file_type::not_found) return ok(std::string(R"({"exists":false})"));

    bool is_dir = std::filesystem::is_directory(st);
    bool is_file = std::filesystem::is_regular_file(st);

    std::string result = "{\"exists\":true,\"type\":\"";
    result += (is_dir ? "dir" : is_file ? "file" : "other");
    result += "\",\"path\":\"";
    utils::escape_json_string(full.string(), result);
    result += '"';
    if (is_file) {
        auto sz = std::filesystem::file_size(full, ec);
        if (!ec) result += ",\"size\":" + std::to_string(sz);
    }
    result += '}';
    return ok(std::move(result));
}

std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx) {
    void *ud = ctx.get();
    return {
        Tool{
            .name = "read_file",
            .description = "Read a file. Relative paths are resolved inside workspace_dir. Text files return their text "
                           "(use offset/max_bytes to page through large files). Images, audio and video files are "
                           "attached to the result so you can look at or listen to them when you support that input.",
            .parameter_schema =
                R"({"type":"object","properties":{"path":{"type":"string"},"max_bytes":{"type":"integer","description":"max bytes to read, default 16384, at most 4194304"},"offset":{"type":"integer","description":"byte offset to start reading at, default 0"}},"required":["path"]})",
            .user_data = ud,
            .parts_callback = read_file_cb},
        Tool{
            .name = "write_file",
            .description = "Write (overwrite) a file. Parent dirs auto-created by default.",
            .parameter_schema =
                R"({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"create_dirs":{"type":"boolean"}},"required":["path","content"]})",
            .callback = write_file_cb,
            .user_data = ud},
        Tool{.name = "append_file",
             .description = "Append text to a file. Creates the file if it does not exist.",
             .parameter_schema =
                 R"({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]})",
             .callback = append_file_cb,
             .user_data = ud},
        Tool{.name = "list_dir",
             .description = "List directory contents. Empty path = workspace root. Returns JSON array.",
             .parameter_schema = R"({"type":"object","properties":{"path":{"type":"string"}},"required":[]})",
             .callback = list_dir_cb,
             .user_data = ud},
        Tool{.name = "create_dir",
             .description = "Create a directory and all parents.",
             .parameter_schema = R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})",
             .callback = create_dir_cb,
             .user_data = ud},
        Tool{.name = "delete_path",
             .description = "Delete a file or directory (recursive). Cannot delete workspace root.",
             .parameter_schema = R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})",
             .callback = delete_path_cb,
             .user_data = ud},
        Tool{.name = "move_path",
             .description = "Move or rename a file or directory.",
             .parameter_schema =
                 R"({"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"}},"required":["from","to"]})",
             .callback = move_path_cb,
             .user_data = ud},
        Tool{.name = "file_info",
             .description = "Get metadata for a path. Returns JSON: {exists, type, size, path}.",
             .parameter_schema = R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})",
             .callback = file_info_cb,
             .user_data = ud},
    };
}

} // namespace agent::tools
