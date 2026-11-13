// Download a model from Hugging Face with the hub library, then list what is downloaded. Needs the network.

#include <agent-cpp/hub.hpp>
#include <cstdio>

int main() {
    agent::hub::PullOptions options;
    options.models_dir = "models";
    options.catalog_path = "models/models.json";
    options.on_progress = [](uint64_t done, uint64_t total, void *) {
        std::fprintf(stderr, "\r  %llu / %llu MiB", static_cast<unsigned long long>(done >> 20),
                     static_cast<unsigned long long>(total >> 20));
        return true;
    };

    auto model = agent::hub::pull_model("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M", options);
    std::fprintf(stderr, "\n");
    if (!model.ok) return std::fprintf(stderr, "pull: %s\n", model.error.message.c_str()), 1;
    std::printf("%s\n  %s (%llu bytes)\n", model.value.name.c_str(), model.value.path.c_str(),
                static_cast<unsigned long long>(model.value.size));

    auto all = agent::hub::list_models(options.catalog_path);
    if (all.ok) std::printf("%zu model(s) in %s\n", all.value.size(), options.catalog_path.c_str());
    return 0;
}
