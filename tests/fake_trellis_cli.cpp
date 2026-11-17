// A stand-in for trellis.cpp's trellis-cli, for the engine tests: it takes the same flags, writes a small file with the GLB magic to --output,
// and records the arguments it got in <output>.args. FAKE_TRELLIS_FAIL=1 makes it fail with a message, FAKE_TRELLIS_SLEEP_MS=N makes it
// take that long first (to test cancellation).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

int main(int argc, char **argv) {
    std::string output, models, image;
    std::string all;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        all += a + "\n";
        if (i + 1 < argc && (a == "--output" || a == "-o")) output = argv[i + 1];
        if (i + 1 < argc && (a == "--models" || a == "-m")) models = argv[i + 1];
        if (i + 1 < argc && (a == "--image" || a == "-i")) image = argv[i + 1];
    }
    if (const char *ms = std::getenv("FAKE_TRELLIS_SLEEP_MS")) std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(ms)));
    if (std::getenv("FAKE_TRELLIS_FAIL")) {
        std::fprintf(stderr, "fake trellis-cli: boom\n");
        return 3;
    }
    if (output.empty() || models.empty() || image.empty()) {
        std::fprintf(stderr, "fake trellis-cli: --image, --output and --models are required\n");
        return 2;
    }
    if (std::FILE *f = std::fopen((output + ".args").c_str(), "wb")) {
        std::fwrite(all.data(), 1, all.size(), f);
        std::fclose(f);
    }
    std::FILE *f = std::fopen(output.c_str(), "wb");
    if (!f) return 4;
    std::fwrite("glTF", 1, 4, f);
    const std::string pad(20 * 1024, '\0');
    std::fwrite(pad.data(), 1, pad.size(), f);
    std::fclose(f);
    std::printf("fake trellis-cli: wrote %s\n", output.c_str());
    return 0;
}
