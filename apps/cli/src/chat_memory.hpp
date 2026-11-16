#pragma once

// The agent's notes (remember / recall / forget) and its search through the saved conversations of a project. The chat registers the tools;
// this file holds the logic so it can be tested with a plain store.

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// Words of at least three letters or digits, lower case, without repeats: what two texts are compared by.
std::vector<std::string> search_words(std::string_view text);

// The note names kept with remember, in the order they were kept.
std::vector<std::string> note_keys(Session &session);

// Keeps a note under `key`; the index of names is updated.
Result<void> note_keep(Session &session, std::string_view key, std::string_view value);

// Removes a note by its exact name, or, without one, the single note whose name or text best matches `text`. The answer says what was
// removed, or why nothing was (no match, or several matches the person has to choose from). Never claims a removal that did not happen:
// `removed` tells the truth.
std::string note_forget(Session &session, std::string_view key, std::string_view text, bool &removed);

struct ChatHit {
    std::string id;
    std::string title;
    std::string snippet; // a line of the best matching message
    int score = 0;       // how many of the search words the conversation contains
};

// Searches the titles and the messages of every saved conversation except `skip_id`. `searched` is the number of conversations that were
// read. Best first, at most `max_hits`. An empty `query` lists the most recent conversations instead.
std::vector<ChatHit> search_chats(Session &session, std::string_view query, std::string_view skip_id, size_t &searched,
                                  size_t max_hits = 5);

// Every note ("note:<key>") and every user and assistant message of the newest `max_chats` saved conversations
// ("chat:<id>:<message number>") with its text: what the memory index is filled from.
// A stored user message without the block of notes the chat put in front of it.
std::string_view without_notes(std::string_view message);
std::vector<std::pair<std::string, std::string>> memory_passages(Session &session, size_t max_chats = 200);

// What the recall tool answers: the note named `key`; otherwise the notes and the earlier chats that match `query` (or `key`), with the
// number of chats searched. With nothing to go on it lists the note names and the latest chats.
std::string recall_text(Session &session, std::string_view key, std::string_view query, std::string_view current_id);

} // namespace agent::cli
