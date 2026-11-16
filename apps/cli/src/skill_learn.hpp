#pragma once

// Skills the agent writes for itself. After a task that took several steps, or after it recovered from a mistake, the agent keeps how
// it was done as a skill (<home>/skills/NAME/SKILL.md), so the next time it follows the steps instead of finding them again. When the
// steps of a skill turn out wrong it repairs the skill: new steps replace the old ones, and what went wrong is kept under "Lessons".

#include <string>
#include <string_view>

namespace agent::cli {

// The text of a SKILL.md: front matter (name, description), the steps, and the lessons (one per line, may be empty).
std::string skill_document(std::string_view name, std::string_view description, std::string_view steps, std::string_view lessons);

// The "## Lessons" lines of an existing SKILL.md (without the heading); empty when there are none.
std::string skill_lessons(std::string_view document);
// The description in the front matter of an existing SKILL.md.
std::string skill_description(std::string_view document);
// The body between the front matter and "## Lessons".
std::string skill_steps(std::string_view document);

// The skill_learn tool: {"name":"deploy-site","description":"...","steps":"1. ...","lesson":"the build needs NODE_ENV=production"}.
// A new skill needs name, description and steps. For an existing one any of description, steps and lesson may be given; a lesson is
// added with `today` (YYYY-MM-DD) in front. Returns the answer for the model; `failed` is set when nothing was written.
std::string skill_learn(const std::string &skills_dir, std::string_view arguments_json, std::string_view today, bool &failed);

} // namespace agent::cli
