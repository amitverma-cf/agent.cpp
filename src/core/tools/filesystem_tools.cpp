#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>
#include <filesystem>
#include <fstream>
#include <simdjson.h>
#include <system_error>

namespace agent::tools {

static std::filesystem::path resolve_path(const std::string &ws, std::string_view rel) {
    std::filesystem::path p(rel);
    return p.is_absolute() ? p : std::filesystem::path(ws) / p;
}

static bool parse_obj(std::string_view args, simdjson::ondemand::parser &parser, simdjson::padded_string &json,
                      simdjson::ondemand::document &doc, simdjson::ondemand::object &obj) {
    json = simdjson::padded_string(args.data(), args.size());
    if (parser.iterate(json).get(doc))
        return false;
    if (doc.get_object().get(obj))
        return false;
    return true;
}

static Result<std::string> read_file_cb(std::string_view args, void *ud) {
    const std::string &ws = *static_cast<const std::string *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj))
        return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv;
    size_t max_bytes = 16384;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error())
            continue;
        if (k.value() == "path") {
            field.value().get_string().get(path_sv);
        } else if (k.value() == "max_bytes") {
            int64_t v;
            if (!field.value().get_int64().get(v))
                max_bytes = static_cast<size_t>(v);
        }
    }
    if (path_sv.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full = resolve_path(ws, path_sv);
    std::error_code ec;
    if (!std::filesystem::exists(full, ec))
        return fail<std::string>(ErrorCode::KeyNotFound, "Not found: " + full.string());

    std::ifstream f(full, std::ios::binary);
    if (!f)
        return fail<std::string>(ErrorCode::FilesystemError, "Cannot open: " + full.string());

    std::string content(max_bytes, '\0');
    f.read(content.data(), static_cast<std::streamsize>(max_bytes));
    content.resize(static_cast<size_t>(f.gcount()));
    if (f.peek() != EOF)
        content += "\n[truncated at " + std::to_string(max_bytes) + " bytes]";

    return ok(std::move(content));
}

static Result<std::string> write_file_cb(std::string_view args, void *ud) {
    const std::string &ws = *static_cast<const std::string *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj))
        return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv, content_sv;
    bool create_dirs = true;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error())
            continue;
        auto kv = k.value();
        if (kv == "path") {
            field.value().get_string().get(path_sv);
        } else if (kv == "content") {
            field.value().get_string().get(content_sv);
        } else if (kv == "create_dirs") {
            bool b;
            if (!field.value().get_bool().get(b))
                create_dirs = b;
        }
    }
    if (path_sv.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full = resolve_path(ws, path_sv);
    if (create_dirs) {
        std::error_code ec;
        std::filesystem::create_directories(full.parent_path(), ec);
    }

    std::ofstream f(full, std::ios::binary | std::ios::trunc);
    if (!f)
        return fail<std::string>(ErrorCode::FilesystemError, "Cannot write: " + full.string());
    f.write(content_sv.data(), static_cast<std::streamsize>(content_sv.size()));

    return ok("wrote " + std::to_string(content_sv.size()) + " bytes to " + full.string());
}

static Result<std::string> append_file_cb(std::string_view args, void *ud) {
    const std::string &ws = *static_cast<const std::string *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj))
        return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv, content_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error())
            continue;
        if (k.value() == "path") {
            field.value().get_string().get(path_sv);
        } else if (k.value() == "content") {
            field.value().get_string().get(content_sv);
        }
    }
    if (path_sv.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full = resolve_path(ws, path_sv);
    std::error_code ec;
    std::filesystem::create_directories(full.parent_path(), ec);

    std::ofstream f(full, std::ios::binary | std::ios::app);
    if (!f)
        return fail<std::string>(ErrorCode::FilesystemError, "Cannot open: " + full.string());
    f.write(content_sv.data(), static_cast<std::streamsize>(content_sv.size()));

    return ok("appended " + std::to_string(content_sv.size()) + " bytes to " + full.string());
}

static Result<std::string> list_dir_cb(std::string_view args, void *ud) {
    const std::string &ws = *static_cast<const std::string *>(ud);
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
                    if (!field.value().get_string().get(sv))
                        path_str = std::string(sv);
                }
            }
        }
    }

    auto dir = path_str.empty() ? std::filesystem::path(ws) : resolve_path(ws, path_str);
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec))
        return fail<std::string>(ErrorCode::KeyNotFound, "Not a directory: " + dir.string());

    std::string result = "[";
    bool first = true;
    for (auto &entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec)
            break;
        if (!first)
            result += ',';
        first = false;
        bool is_dir = entry.is_directory(ec);
        std::string name = entry.path().filename().string();
        result += "{\"name\":\"";
        utils::escape_json_string(name, result);
        result += "\",\"type\":\"";
        result += is_dir ? "dir" : "file";
        result += '"';
        if (!is_dir) {
            auto sz = entry.file_size(ec);
            if (!ec)
                result += ",\"size\":" + std::to_string(sz);
        }
        result += '}';
    }
    result += ']';
    return ok(std::move(result));
}

static Result<std::string> create_dir_cb(std::string_view args, void *ud) {
    const std::string &ws = *static_cast<const std::string *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj))
        return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (!k.error() && k.value() == "path")
            field.value().get_string().get(path_sv);
    }
    if (path_sv.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full = resolve_path(ws, path_sv);
    std::error_code ec;
    std::filesystem::create_directories(full, ec);
    if (ec)
        return fail<std::string>(ErrorCode::FilesystemError, "Failed: " + ec.message());

    return ok("created " + full.string());
}

static Result<std::string> delete_path_cb(std::string_view args, void *ud) {
    const std::string &ws = *static_cast<const std::string *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj))
        return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (!k.error() && k.value() == "path")
            field.value().get_string().get(path_sv);
    }
    if (path_sv.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full = resolve_path(ws, path_sv);
    std::error_code ec;
    auto count = std::filesystem::remove_all(full, ec);
    if (ec)
        return fail<std::string>(ErrorCode::FilesystemError, "Failed: " + ec.message());

    return ok("deleted " + std::to_string(count) + " entries at " + full.string());
}

static Result<std::string> move_path_cb(std::string_view args, void *ud) {
    const std::string &ws = *static_cast<const std::string *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj))
        return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view from_sv, to_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error())
            continue;
        if (k.value() == "from")
            field.value().get_string().get(from_sv);
        else if (k.value() == "to")
            field.value().get_string().get(to_sv);
    }
    if (from_sv.empty() || to_sv.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'from' or 'to'");

    auto src = resolve_path(ws, from_sv);
    auto dst = resolve_path(ws, to_sv);
    std::error_code ec;
    std::filesystem::create_directories(dst.parent_path(), ec);
    std::filesystem::rename(src, dst, ec);
    if (ec)
        return fail<std::string>(ErrorCode::FilesystemError, "Failed: " + ec.message());

    return ok("moved " + src.string() + " -> " + dst.string());
}

static Result<std::string> file_info_cb(std::string_view args, void *ud) {
    const std::string &ws = *static_cast<const std::string *>(ud);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parse_obj(args, parser, json, doc, obj))
        return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    std::string_view path_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (!k.error() && k.value() == "path")
            field.value().get_string().get(path_sv);
    }
    if (path_sv.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'path'");

    auto full = resolve_path(ws, path_sv);
    std::error_code ec;
    auto st = std::filesystem::status(full, ec);

    if (ec || st.type() == std::filesystem::file_type::not_found)
        return ok(std::string(R"({"exists":false})"));

    bool is_dir = std::filesystem::is_directory(st);
    bool is_file = std::filesystem::is_regular_file(st);

    std::string result = "{\"exists\":true,\"type\":\"";
    result += (is_dir ? "dir" : is_file ? "file" : "other");
    result += "\",\"path\":\"";
    utils::escape_json_string(full.string(), result);
    result += '"';
    if (is_file) {
        auto sz = std::filesystem::file_size(full, ec);
        if (!ec)
            result += ",\"size\":" + std::to_string(sz);
    }
    result += '}';
    return ok(std::move(result));
}

std::vector<Tool> get_filesystem_tools(const std::string *workspace_dir) {
    void *ud = const_cast<void *>(static_cast<const void *>(workspace_dir));
    return {
        Tool{
            .name = "read_file",
            .description = "Read a file. Relative paths are resolved inside workspace_dir.",
            .parameter_schema =
                R"({"type":"object","properties":{"path":{"type":"string"},"max_bytes":{"type":"integer","description":"max bytes to read, default 16384"}},"required":["path"]})",
            .callback = read_file_cb,
            .user_data = ud},
        Tool{
            .name = "write_file",
            .description = "Write (overwrite) a file. Parent dirs auto-created by default.",
            .parameter_schema =
                R"({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"create_dirs":{"type":"boolean"}},"required":["path","content"]})",
            .callback = write_file_cb,
            .user_data = ud},
        Tool{
            .name = "append_file",
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
             .description = "Delete a file or directory (recursive). Use with care.",
             .parameter_schema = R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})",
             .callback = delete_path_cb,
             .user_data = ud},
        Tool{
            .name = "move_path",
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
