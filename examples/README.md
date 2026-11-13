# Examples

Each example is one short program with no arguments: the model files and the engine are named at the top of `main()`, so the code shows
exactly what is needed. They use real models; none of them uses a mock.

```sh
cmake -B build -G Ninja                                                       # every engine is built by default
cmake --build build
cmake --install build --prefix ~/agent-cpp

cmake -S examples -B build-examples -DCMAKE_PREFIX_PATH=~/agent-cpp          # the examples as a project of their own
cmake --build build-examples                                                   # copies the engines each example uses into build-examples/engines/
examples/download-models.sh                                                    # once; run examples from this folder
./build-examples/hello
```

| Example | What it shows | Needs |
|---|---|---|
| [`hello`](01-hello-agent) | open a model, ask a question, stream the answer | llama, Qwen2.5-0.5B |
| [`tools`](02-tools) | a tool of your own and the permission gate | llama, Qwen2.5-0.5B |
| [`graph`](03-graph-workflow) | two agents in a row; `graphs/` holds the same idea as JSON for `agent run` | llama, Qwen2.5-0.5B |
| [`engine_host`](04-custom-engine) | write an engine plugin and drive it in process and under process isolation | nothing (builds its own plugin) |
| [`rag`](05-rag-over-docs) | index a folder, retrieve, answer from the excerpts (run from the repository root) | llama, bge-small, Qwen2.5-0.5B |
| [`prompt_to_image`](06-prompt-to-image) | a chat model writes a prompt, a stable-diffusion model draws it | llama and sdcpp |
| [`pull`](07-model-pull) | download a Hugging Face model with the hub library | network |
| [`08-engine-plugin-template`](08-engine-plugin-template) | a standalone CMake project for an out-of-tree engine plugin | an installed agent.cpp |
| [`draw`](09-image-generation) | draw an image | sdcpp, SDXS |
| [`speech`](10-speech) | speak a sentence and transcribe it back | audiocpp, Parakeet, Pocket TTS |
