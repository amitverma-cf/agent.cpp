# Decisions: choice, score, yes/no

`decide`, `decide_score` and `decide_yes_no` ask a model a question with a fixed set of answers and
return a probability distribution over them, computed from the model's log-probabilities instead of
parsing generated text. Use them for routing ("which team handles this ticket?"), grading ("rate this
answer 1 to 5"), guards ("is this request about billing?") and cheap classifiers inside graphs.

They run the session's engine with `AGENT_CAP_DECIDE`. The [llama engine](engines.md#decisions-choice--score--yes-no)
implements it for any instruct GGUF (and for a vision model with `mmproj`, with images).

```cpp
#include <agent-cpp/agent.hpp>

agent::Config cfg;
cfg.provider       = agent::AiProvider::Engine;
cfg.engine         = "llama";
cfg.model          = ".workspace/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
cfg.context_window = 1024;
auto session = agent::init(cfg);

const std::string_view teams[] = {"billing", "shipping", "technical"};
auto route = agent::decide(session.value, "Customer: I was charged twice for my order.", teams);
// route.value.options[route.value.choice] == "billing", route.value.probs = {0.97, 0.02, 0.01}

auto urgent = agent::decide_yes_no(session.value, "Customer: the site is down for everyone. Is this urgent?");
// urgent.value = P(yes)

auto grade = agent::decide_score(session.value, "How polite is: 'fix it now, idiots'?", 1, 5);
// grade.value.score = expected value on 1..5, grade.value.probs = one entry per level
```

## API

```cpp
struct DecideOptions {
    std::span<const Media> media; // images for a vision model (engine option "mmproj")
    std::string_view system;      // optional system message
    bool instructions = true;     // let the engine append the answer format to the question
};

struct Decision {
    std::vector<std::string> options; // in request order ("Yes","No" for yes/no; "1".."5" for a scale)
    std::vector<float> probs;         // one per option, sums to 1
    size_t choice = 0;                // index of the most likely option
    float confidence = 0.0f;          // probs[choice]
    float score = 0.0f;               // decide_score: expected value; decide_yes_no: P(yes); decide: confidence
};

Result<Decision> decide(Session &session, std::string_view question, std::span<const std::string_view> options,
                        const DecideOptions &opts = {});
Result<Decision> decide_score(Session &session, std::string_view question, int min = 1, int max = 5,
                              const DecideOptions &opts = {});
Result<float> decide_yes_no(Session &session, std::string_view question, const DecideOptions &opts = {});
```

- `decide` needs at least two non-empty options; `decide_score` needs `min < max` and at most 101
  levels. Violations fail with `InvalidConfig`; an engine without `AGENT_CAP_DECIDE` fails with
  `UnsupportedProvider`.
- Like `embed` / `rerank`, the request and response live in a per-thread arena, not the session's
  per-turn arena, so these can be called from inside a tool or a graph Function node of another
  session's turn. Don't call them concurrently with `run_turn` on the same session from several threads.
- `instructions = false` sends the question verbatim; use it when the question already lists the
  options and says how to answer.

## How the llama engine scores

1. The question (with `system` and images) is rendered with the model's chat template, with an answer
   format appended to the user message: the option list and "Answer with exactly one of the options,
   word for word." (choice), "Answer with Yes or No." (yes/no) or "Answer with a single whole number
   from 1 to 5." (score).
2. The prompt is prefilled once. Each option is then scored as the assistant's reply: the sum of the
   log-probabilities of its tokens followed by the end-of-turn token (so `"1"` does not also absorb
   the mass of `"10"`, and `"a cat"` does not absorb `"a cathedral"`). Every option is decoded after
   the prompt and removed from the KV cache again, so one sequence slot serves all options.
3. The sums are normalized with a softmax. `decide_score` reports the expected value
   `sum(p_i * level_i)`, which is smoother than the arg-max level.

Cost: one prompt prefill plus one small decode per option (options of one token cost nothing beyond
the end-of-turn token). The model never generates, so temperature and `max_tokens` don't apply.

Scores reflect what a small model believes; calibrate thresholds on your own data. A 0.5B model is
fine for easy routing and yes/no facts, less so for nuanced grading.

## Native decision models (System One)

llama.cpp also ships dedicated *decision models* (Julia-1, Laya, Kev-4B, lev, Clef, ...; GGUFs with an
`<arch>.decision.type` key) served by llama-server's `/v1/systemone` endpoint. Each type reads its
answers from a model-specific head or label token with its own prompt layout and calibration (about
1.8k lines in `tools/server/server-decision.cpp`, not part of llama.cpp's libraries). agent.cpp does
not reproduce that yet: the llama engine detects such a model and fails `AGENT_CAP_DECIDE` with
`AGENT_ERR_UNSUPPORTED` and a message pointing to llama-server. Use a general instruct model here.

## Engine protocol (`AGENT_CAP_DECIDE`)

For engine authors and direct `run_task` callers. The request's messages carry the question (TEXT
parts, optional image FILE/BYTES parts); the answers come in `params_json` (through the core:
`InferRequest::params_json`, which is merged into the core's own parameters):

| `params_json` | Options scored |
|---|---|
| `{"mode":"choice","options":["billing","shipping"]}` | the given strings (2 to 256) |
| `{"mode":"score","scale":[1,5]}` | `"1"` .. `"5"` (default scale `[1,5]`, at most 101 levels) |
| `{"mode":"yesno"}` | `"Yes"`, `"No"` |

Optional keys: `"instructions": false` (don't append the answer format), `"end": false` (don't add
the end-of-turn token to each option).

The response has a TEXT part with a JSON summary and an f32 TENSOR part `"probs"` (`[n_options]`, in
option order):

```json
{"mode":"score","choice":"5","index":4,"confidence":0.81,"score":4.77,"scale":[1,5],
 "options":["1","2","3","4","5"],"probs":[0.001,0.002,0.03,0.16,0.81],"logprobs":[-9.1,-8.4,-5.8,-4.1,-2.5]}
```

`"yes"` (P(yes)) replaces `"score"` in yes/no mode. `logprobs` are the raw per-option sums before
normalization.
