# Decisions: choice, score, yes/no

`#include <agent-cpp/agent.hpp>`

`decide`, `decide_score` and `decide_yes_no` ask a model a question with a fixed set of answers. They return a probability distribution
over the answers, computed from the model's log-probabilities, so no text is parsed and nothing is generated. Use them for routing
("which team handles this ticket?"), grading ("rate this answer 1 to 5"), guards ("is this request about billing?") and cheap classifiers
inside graphs.

They run on the session's engine with the `AGENT_CAP_DECIDE` capability. The [llama engine](engines.md#decisions-choice--score--yes-no)
implements it for any instruct GGUF, and for a vision model with `mmproj` when images are passed.

```cpp
#include <agent-cpp/agent.hpp>

agent::Config base;
base.context_window = 1024;
auto session = agent::open_session(agent::engines::llama, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", base);

const std::string_view teams[] = {"billing", "shipping", "technical"};
auto route = agent::decide(session.value, "Customer: I was charged twice for my order.", teams);
// route.value.options[route.value.choice] == "billing", route.value.probs = {0.97, 0.02, 0.01}

auto urgent = agent::decide_yes_no(session.value, "Customer: the site is down for everyone. Is this urgent?");
// urgent.value = P(yes)

auto grade = agent::decide_score(session.value, "How polite is: 'fix it now, idiots'?", 1, 5);
// grade.value.score = expected value on 1..5, grade.value.probs = one entry per level
```

Error checks are left out of the examples. Check `.ok` on each result.

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

- `decide` needs at least two options. `decide_score` needs `min < max` and at most 101 levels. Violations fail with
  `InvalidConfig`. A session whose engine lacks `AGENT_CAP_DECIDE` fails with `UnsupportedProvider`.
- `decide_yes_no` returns the probability of "Yes" directly, as `Result<float>`.
- Like `embed` and `rerank`, these run outside the session's per-turn arena. They can be called from inside a tool or a graph Function
  node of another session's turn. Do not call them on one session from several threads at once, or while `run_turn` runs on that session.
- `instructions = false` sends the question as it is. Use it when the question already lists the options and says how to answer.
- With images, pass them in `DecideOptions::media` and open the session with the `mmproj` option.

```cpp
// Routing with an image, using a vision model (the mmproj role names its projector).
auto vision_session = agent::open_session(agent::engines::llama,
                                          {{"model", "models/qwen2.5-vl-3b-instruct-q4_k_m.gguf"}, {"mmproj", "models/mmproj-f16.gguf"}});
agent::Media photo[] = {{"photos/receipt.jpg", "image/jpeg"}};
agent::DecideOptions opts;
opts.media = photo;
const std::string_view kinds[] = {"receipt", "invoice", "other"};
auto kind = agent::decide(vision_session.value, "What kind of document is this?", kinds, opts);
```

## How the llama engine scores

1. The question, with `system` and any images, is rendered with the model's chat template. The answer format is appended to the user
   message: the option list and "Answer with exactly one of the options, word for word." (choice), "Answer with Yes or No." (yes/no), or
   "Answer with a single whole number from 1 to 5." (score).
2. The prompt is prefilled once. Each option is then scored as the assistant's reply: the sum of the log-probabilities of its tokens,
   followed by the end-of-turn token. So `"1"` does not absorb the mass of `"10"`, and `"a cat"` does not absorb `"a cathedral"`. Each
   option is decoded after the prompt and then removed from the KV cache, so one sequence slot serves all options.
3. The sums are normalized with a softmax. `decide_score` reports the expected value `sum(p_i * level_i)`, which is smoother than the
   most likely level.

Cost: one prompt prefill and one short decode per option. Options of one token cost only the end-of-turn token. The model never generates
text, so `temperature` and `max_tokens` do not apply.

Scores reflect what a small model believes. Calibrate thresholds on your own data. A 0.5B model is fine for simple routing and yes/no
facts, and weaker for nuanced grading.

## Native decision models (System One)

llama.cpp also has dedicated decision models: GGUFs with an `<arch>.decision.type` key. They read their answers from model-specific
heads and prompt layouts. The llama engine detects one when it loads the file. From then on, `AGENT_CAP_DECIDE` requests on that
session go to llama.cpp's own System One handler, in process. Choice and yes/no questions are sent as a choice question over the
options, and score questions as a score question over the levels. The probabilities come from the model, not from option scoring, so
its calibration applies.

This path has not been tested with a real decision model in this project. Verify the results on your own data before you rely on them.
Other GGUFs use the option scoring described above.

## Engine protocol (`AGENT_CAP_DECIDE`)

For engine authors and direct `run_task` callers. The request's messages carry the question as TEXT parts, with optional image FILE or
BYTES parts. The answers come in `params_json`, which the core merges into the request parameters:

| `params_json` | Options scored |
|---|---|
| `{"mode":"choice","options":["billing","shipping"]}` | the given strings (2 to 256) |
| `{"mode":"score","scale":[1,5]}` | `"1"` .. `"5"` (default scale `[1,5]`, at most 101 levels) |
| `{"mode":"yesno"}` | `"Yes"`, `"No"` |

Optional keys: `"instructions": false` (do not append the answer format) and `"end": false` (do not add the end-of-turn token to each
option).

The response has a TEXT part with a JSON summary and an f32 TENSOR part named `"probs"` (shape `[n_options]`, in option order):

```json
{"mode":"score","choice":"5","index":4,"confidence":0.81,"score":4.77,"scale":[1,5],
 "options":["1","2","3","4","5"],"probs":[0.001,0.002,0.03,0.16,0.81],"logprobs":[-9.1,-8.4,-5.8,-4.1,-2.5]}
```

In yes/no mode, `"yes"` (P(yes)) replaces `"score"`. `logprobs` are the raw per-option sums before normalization.

See also [rag.md](rag.md) for embeddings and reranking, and [api.md](../api.md) for the other one-shot tasks.
