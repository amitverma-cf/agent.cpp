#include "../utils/utils.hpp"
#include "hal/path.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <simdjson.h>
#include <sstream>

namespace agent {

namespace {

// Parses the minimal SKILL.md frontmatter block:
//   ---
//   name: skill-name
//   description: one-line description
//   ---
// Only `name`/`description` are extracted; a full YAML parser isn't needed for two keys.
struct Frontmatter {
    std::string name;
    std::string description;
};

std::string_view trim(std::string_view s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

bool parse_frontmatter(std::string_view content, Frontmatter &out) {
    if (!content.starts_with("---")) return false;
    size_t body_start = content.find('\n', 3);
    if (body_start == std::string_view::npos) return false;
    size_t end = content.find("\n---", body_start);
    if (end == std::string_view::npos) return false;

    std::string_view block = content.substr(body_start + 1, end - body_start - 1);
    size_t pos = 0;
    while (pos < block.size()) {
        size_t line_end = block.find('\n', pos);
        if (line_end == std::string_view::npos) line_end = block.size();
        std::string_view line = block.substr(pos, line_end - pos);
        size_t colon = line.find(':');
        if (colon != std::string_view::npos) {
            std::string_view key = trim(line.substr(0, colon));
            std::string_view value = trim(line.substr(colon + 1));
            // YAML scalars may be quoted: name: "pdf-tools"
            if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') && value.back() == value.front())
                value = value.substr(1, value.size() - 2);
            if (key == "name") out.name = std::string(value);
            else if (key == "description") out.description = std::string(value);
        }
        pos = line_end + 1;
    }
    return !out.name.empty() && !out.description.empty();
}

std::string_view skill_media_mime(const std::filesystem::path &p) {
    std::string ext = p.extension().string();
    for (char &c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".gif") return "image/gif";
    if (ext == ".webp") return "image/webp";
    if (ext == ".wav") return "audio/wav";
    if (ext == ".mp3") return "audio/mpeg";
    return {};
}

// {"name": ..., "file"?: ...}: the skill's SKILL.md, or one of the files bundled in its directory
// (scripts, references, templates; images / audio come back as media). "file" must stay inside the
// skill directory.
Result<ToolResult> skill_loader_callback(std::string_view arguments, void *user_data) {
    auto *session_cell = static_cast<Session **>(user_data);
    Session *session = (session_cell && *session_cell) ? *session_cell : nullptr;
    if (!session) return fail<ToolResult>(ErrorCode::InvalidConfig, "use_skill: session reference is null");

    std::string_view name_sv, file_sv;
    simdjson::ondemand::parser parser;
    simdjson::padded_string json(arguments.data(), arguments.size());
    simdjson::ondemand::document doc;
    if (!parser.iterate(json).get(doc)) {
        simdjson::ondemand::object obj;
        if (!doc.get_object().get(obj)) {
            for (auto field : obj) {
                auto k = field.unescaped_key();
                if (k.error()) continue;
                if (k.value() == "name") {
                    if (field.value().get_string().get(name_sv)) name_sv = {};
                } else if (k.value() == "file") {
                    if (field.value().get_string().get(file_sv)) file_sv = {};
                }
            }
        }
    }
    if (name_sv.empty()) return fail<ToolResult>(ErrorCode::InvalidConfig, "use_skill: missing 'name'");

    ToolResult out;
    if (file_sv.empty()) {
        auto body = read_skill_body(*session, name_sv);
        if (!body.ok) return fail<ToolResult>(body.error.code, body.error.message);
        auto it = session->skills.find(std::string(name_sv));
        out.text = "Skill directory: " + it->second.path + " (load bundled files with use_skill's \"file\")\n\n" + body.value;
        return ok(std::move(out));
    }

    auto it = session->skills.find(std::string(name_sv));
    if (it == session->skills.end()) return fail<ToolResult>(ErrorCode::KeyNotFound, "Unknown skill: " + std::string(name_sv));
    auto resolved = hal::resolve_under_workspace(it->second.path, file_sv, true);
    if (!resolved.ok) return fail<ToolResult>(resolved.error.code, "use_skill: 'file' must be inside the skill directory");
    const std::filesystem::path &path = resolved.value;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        return fail<ToolResult>(ErrorCode::KeyNotFound, "use_skill: no such file: " + path.string());
    if (std::string_view mime = skill_media_mime(path); !mime.empty()) {
        out.text = path.filename().string() + " attached";
        out.media.push_back(Media{.path = path.string(), .mime = std::string(mime)});
        return ok(std::move(out));
    }
    constexpr size_t kMaxFile = 256 * 1024;
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail<ToolResult>(ErrorCode::FilesystemError, "Cannot open: " + path.string());
    std::string content(kMaxFile + 1, '\0');
    f.read(content.data(), static_cast<std::streamsize>(content.size()));
    content.resize(static_cast<size_t>(std::max<std::streamsize>(f.gcount(), 0)));
    const bool more = content.size() > kMaxFile;
    if (more) content.resize(utils::utf8_prefix_length(content, kMaxFile));
    if (content.find('\0') != std::string::npos) {
        out.text = "[binary file " + path.filename().string() + "; not shown as text]";
        return ok(std::move(out));
    }
    utils::sanitize_utf8(content);
    out.text = std::move(content);
    if (more) out.text += "\n[truncated at " + std::to_string(kMaxFile) + " bytes]";
    return ok(std::move(out));
}

} // namespace

Result<void> load_skills_dir(Session &session, std::string_view skills_dir) {
    std::error_code ec;
    if (!std::filesystem::is_directory(skills_dir, ec))
        return fail(ErrorCode::InvalidConfig, "skills_dir is not a directory: " + std::string(skills_dir));

    // Range-based for would use directory_iterator::operator++() (the throwing overload) on
    // every iteration, even though construction here is non-throwing -- advance manually with
    // the error_code overload so a mid-scan filesystem error can't throw an uncaught
    // std::filesystem::filesystem_error.
    for (auto it = std::filesystem::directory_iterator(skills_dir, ec); !ec && it != std::filesystem::directory_iterator();
         it.increment(ec)) {
        const auto &entry = *it;
        std::error_code entry_ec; // the non-throwing overloads: no exceptions in the core
        if (!entry.is_directory(entry_ec)) continue;

        std::filesystem::path skill_md = entry.path() / "SKILL.md";
        if (!std::filesystem::is_regular_file(skill_md, entry_ec)) continue;

        std::ifstream f(skill_md, std::ios::binary);
        if (!f) continue;
        std::stringstream buf;
        buf << f.rdbuf();
        std::string content = buf.str();

        Frontmatter fm;
        if (!parse_frontmatter(content, fm)) continue;

        // Absolute, so use_skill's "file" and the path shown to the model don't depend on the cwd.
        std::filesystem::path dir = std::filesystem::absolute(entry.path(), entry_ec);
        if (entry_ec) dir = entry.path();
        session.skills[fm.name] = Skill{.name = fm.name, .description = fm.description, .path = dir.lexically_normal().string()};
    }

    if (session.session_ptr_cell) {
        bool already_registered = false;
        for (const auto &t : session.config.tools)
            if (t.name == "use_skill") already_registered = true;

        if (!already_registered) {
            session.config.tools.push_back(Tool{
                .name = "use_skill",
                .description = "Load the full instructions for a named skill. Call this before "
                               "using a skill whose name matches the task at hand. Optional \"file\": a file "
                               "bundled with the skill (path relative to the skill directory) to load instead.",
                .parameter_schema =
                    R"({"type":"object","properties":{"name":{"type":"string"},"file":{"type":"string","description":"file inside the skill directory, e.g. references/forms.md"}},"required":["name"]})",
                .user_data = session.session_ptr_cell.get(),
                .parts_callback = skill_loader_callback,
            });
        }
    }

    rebuild_tool_index(session);
    return ok();
}

Result<std::string> read_skill_body(Session &session, std::string_view name) {
    auto it = session.skills.find(std::string(name));
    if (it == session.skills.end()) return fail<std::string>(ErrorCode::KeyNotFound, "Unknown skill: " + std::string(name));

    std::filesystem::path skill_md = std::filesystem::path(it->second.path) / "SKILL.md";
    std::ifstream f(skill_md, std::ios::binary);
    if (!f) return fail<std::string>(ErrorCode::FilesystemError, "Cannot open: " + skill_md.string());

    std::stringstream buf;
    buf << f.rdbuf();
    return ok(buf.str());
}

} // namespace agent
