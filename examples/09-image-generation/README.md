# 09 image generation

`agent::open_session` with a checkpoint (the `model` role, plus an optional `vae`), then `agent::draw`. Runs on the stable-diffusion.cpp engine
(`-DAGENT_ENGINE_SDCPP=ON`) with a model such as `agent model pull sdxs-512`. The smoke test runs the same code on the mock engine.
