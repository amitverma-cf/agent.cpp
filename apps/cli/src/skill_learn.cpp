#include "skill_learn.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <simdjson.h>

namespace agent::cli {

namespace {

constexpr std::string_view kLessons = "\n## Lessons\n";

std::string trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return std::string(s);
}

std::string one_line(std::string_view s) {
    std::string r = trim(s);
    for (char &c : r)
        if (c == '\n' || c == '\r') c = ' ';
    return r;
}

// After the closing "---" of the front matter; 0 when there is none.
size_t body_start(std::string_view doc) {
    if (doc.substr(0, 3) != "---") return 0;
    const size_t end = doc.find("\n---", 3);
    if (end == std::string_view::npos) return 0;
    const size_t line_end = doc.find('\n', end + 4);
    return line_end == std::string_view::npos ? doc.size() : line_end + 1;
}

} // namespace

std::string skill_document(std::string_view name, std::string_view description, std::string_view steps, std::string_view lessons) {
    std::string doc = "---\nname: " + std::string(name) + "\ndescription: " + one_line(description) + "\n---\n\n" + trim(steps) + "\n";
    if (const std::string l = trim(lessons); !l.empty()) doc += std::string(kLessons) + l + "\n";
    return doc;
}

std::string skill_lessons(std::string_view document) {
    const size_t at = document.find(kLessons);
    return at == std::string_view::npos ? std::string() : trim(document.substr(at + kLessons.size()));
}

std::string skill_description(std::string_view document) {
    const size_t end = body_start(document);
    const std::string_view front = document.substr(0, end);
    const size_t at = front.find("\ndescription:");
    if (at == std::string_view::npos) return {};
    const size_t from = at + 13;
    const size_t line_end = front.find('\n', from);
    return trim(front.substr(from, line_end == std::string_view::npos ? std::string_view::npos : line_end - from));
}

std::string skill_steps(std::string_view document) {
    const size_t from = body_start(document);
    const size_t to = document.find(kLessons, from);
    return trim(document.substr(from, to == std::string_view::npos ? std::string_view::npos : to - from));
}

std::string skill_learn(const std::string &skills_dir, std::string_view arguments_json, std::string_view today, bool &failed) {
    failed = true;
    simdjson::dom::parser parser;
    simdjson::padded_string padded(arguments_json);
    simdjson::dom::element doc;
    std::string_view name, description, steps, lesson;
    if (parser.parse(padded).get(doc) || doc["name"].get(name)) return "skill_learn needs a name";
    if (doc["description"].get(description)) description = "";
    if (doc["steps"].get(steps)) steps = "";
    if (doc["lesson"].get(lesson)) lesson = "";
    if (name.empty() || name.size() > 64) return "the skill name must be 1 to 64 characters";
    for (const char c : name)
        if (!(std::islower(static_cast<unsigned char>(c)) || std::isdigit(static_cast<unsigned char>(c)) || c == '-'))
            return "the skill name may have lower-case letters, digits and - only, for example deploy-site";

    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::path(skills_dir) / std::string(name);
    const fs::path file = dir / "SKILL.md";
    std::string old;
    if (fs::is_regular_file(file, ec)) {
        std::ifstream in(file, std::ios::binary);
        old.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    const bool is_new = old.empty();
    if (is_new && (trim(description).empty() || trim(steps).empty()))
        return "a new skill needs a description (when to use it, one line) and the steps";
    if (!is_new && trim(description).empty() && trim(steps).empty() && trim(lesson).empty())
        return "nothing to change: give new steps, a new description or a lesson";

    const std::string final_description = trim(description).empty() ? skill_description(old) : std::string(description);
    const std::string final_steps = trim(steps).empty() ? skill_steps(old) : std::string(steps);
    std::string lessons = skill_lessons(old);
    if (const std::string l = one_line(lesson); !l.empty())
        lessons += (lessons.empty() ? "" : "\n") + std::string("- ") + std::string(today) + ": " + l;

    fs::create_directories(dir, ec);
    const fs::path tmp = dir / "SKILL.md.tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return "the skill could not be written to " + file.string();
        out << skill_document(name, final_description, final_steps, lessons);
    }
    fs::rename(tmp, file, ec);
    if (ec) return "the skill could not be written to " + file.string();
    failed = false;
    if (is_new)
        return "learned the skill '" + std::string(name) + "' (" + file.string() +
               "); it is used the next time a request fits its description";
    return "updated the skill '" + std::string(name) + "'" + (trim(steps).empty() ? "" : ": new steps") +
           (trim(lesson).empty() ? "" : std::string(trim(steps).empty() ? ": " : ", ") + "lesson kept");
}

} // namespace agent::cli
