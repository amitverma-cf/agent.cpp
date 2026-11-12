// The hub's HTTP client against a throw-away server on the loopback interface (test code only: the product opens no listening socket).
#include <agent-cpp/hub.hpp>
#include <atomic>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>
#include <httplib.h>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using namespace agent::hub;

namespace {

struct TestServer {
    httplib::Server svr;
    std::thread th;
    int port = 0;
    TestServer() = default;
    void start() {
        port = svr.bind_to_any_port("127.0.0.1");
        th = std::thread([this] { svr.listen_after_bind(); });
        svr.wait_until_ready();
    }
    std::string url(const std::string &path) const { return "http://127.0.0.1:" + std::to_string(port) + path; }
    ~TestServer() {
        svr.stop();
        if (th.joinable()) th.join();
    }
};

HttpOptions options() {
    HttpOptions o;
    o.connect_timeout_sec = 5;
    o.read_timeout_sec = 10;
    return o;
}

} // namespace

TEST_CASE("hub http: GET returns the body, follows relative and absolute redirects", "[hub_http]") {
    TestServer other, s;
    other.svr.Get("/final", [](const httplib::Request &, httplib::Response &res) { res.set_content("from the other host", "text/plain"); });
    other.start();
    s.svr.Get("/plain", [](const httplib::Request &, httplib::Response &res) { res.set_content("hello", "text/plain"); });
    s.svr.Get("/rel", [](const httplib::Request &, httplib::Response &res) { res.set_redirect("/plain", 302); });
    const std::string target = other.url("/final");
    s.svr.Get("/abs", [target](const httplib::Request &, httplib::Response &res) { res.set_redirect(target, 302); });
    s.svr.Get("/loop", [](const httplib::Request &, httplib::Response &res) { res.set_redirect("/loop", 302); });
    s.start();

    auto plain = http_get_text(s.url("/plain"), options());
    REQUIRE(plain.ok);
    CHECK(plain.value == "hello");
    auto rel = http_get_text(s.url("/rel"), options());
    REQUIRE(rel.ok);
    CHECK(rel.value == "hello");
    auto abs = http_get_text(s.url("/abs"), options());
    REQUIRE(abs.ok);
    CHECK(abs.value == "from the other host");

    HttpOptions few = options();
    few.max_redirects = 3;
    auto loop = http_get_text(s.url("/loop"), few);
    CHECK_FALSE(loop.ok);
    CHECK(loop.error.message.find("redirect") != std::string::npos);
}

TEST_CASE("hub http: error statuses become readable errors", "[hub_http]") {
    TestServer s;
    s.svr.Get("/missing", [](const httplib::Request &, httplib::Response &res) {
        res.status = 404;
        res.set_content("nope", "text/plain");
    });
    s.svr.Get("/gated", [](const httplib::Request &, httplib::Response &res) { res.status = 403; });
    s.svr.Get("/proxy", [](const httplib::Request &, httplib::Response &res) { res.status = 407; });
    s.start();
    auto missing = http_get_text(s.url("/missing"), options());
    REQUIRE_FALSE(missing.ok);
    CHECK(missing.error.message.find("404") != std::string::npos);
    CHECK(missing.error.message.find("not found") != std::string::npos);
    auto gated = http_get_text(s.url("/gated"), options());
    REQUIRE_FALSE(gated.ok);
    CHECK(gated.error.message.find("HF_TOKEN") != std::string::npos);
    auto proxy = http_get_text(s.url("/proxy"), options());
    REQUIRE_FALSE(proxy.ok);
    CHECK(proxy.error.message.find("proxy") != std::string::npos);
    // Nothing listens there: a network error, not a crash.
    auto down = http_get_text("http://127.0.0.1:1/", options());
    CHECK_FALSE(down.ok);
}

TEST_CASE("hub http: a transient 400 is retried", "[hub_http]") {
    TestServer s;
    std::atomic<int> calls{0};
    s.svr.Get("/flaky", [&calls](const httplib::Request &, httplib::Response &res) {
        if (calls++ == 0) {
            res.status = 400;
            return;
        }
        res.set_content("second time lucky", "text/plain");
    });
    s.start();
    auto r = http_get_text(s.url("/flaky"), options());
    REQUIRE(r.ok);
    CHECK(r.value == "second time lucky");
    CHECK(calls.load() == 2);
}

TEST_CASE("hub http: a body over the limit fails instead of being cut", "[hub_http]") {
    TestServer s;
    s.svr.Get("/big", [](const httplib::Request &, httplib::Response &res) { res.set_content(std::string(5000, 'x'), "text/plain"); });
    s.start();
    auto r = http_get_text(s.url("/big"), options(), 1000);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.message.find("larger than") != std::string::npos);
    auto fits = http_get_text(s.url("/big"), options(), 6000);
    REQUIRE(fits.ok);
    CHECK(fits.value.size() == 5000);
}

TEST_CASE("hub http: download_file writes, verifies size and resumes a partial file", "[hub_http]") {
    TestServer s;
    std::string content;
    for (int i = 0; i < 20000; ++i) content += static_cast<char>('a' + i % 26);
    s.svr.Get("/file",
              [&content](const httplib::Request &, httplib::Response &res) { res.set_content(content, "application/octet-stream"); });
    s.start();

    const fs::path dir = fs::temp_directory_path() / "agent_test_hub_http";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    const std::string dest = (dir / "out.bin").string();

    DownloadOptions opts;
    opts.expected_size = content.size();
    auto ok_dl = download_file(s.url("/file"), dest, options(), opts);
    REQUIRE(ok_dl.ok);
    {
        std::ifstream in(dest, std::ios::binary); // closed before the file is removed below (Windows cannot delete an open file)
        std::string got((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(got == content);
    }
    CHECK_FALSE(fs::exists(dest + ".part"));

    // A wrong expected size is refused and leaves no final file.
    fs::remove(dest, ec);
    DownloadOptions wrong;
    wrong.expected_size = content.size() + 5;
    auto bad = download_file(s.url("/file"), dest, options(), wrong);
    CHECK_FALSE(bad.ok);
    CHECK_FALSE(fs::exists(dest));

    // A wrong checksum is refused too.
    fs::remove(dest + ".part", ec);
    DownloadOptions sum;
    sum.expected_sha256 = std::string(64, '0');
    auto bad_sum = download_file(s.url("/file"), dest, options(), sum);
    CHECK_FALSE(bad_sum.ok);
    CHECK_FALSE(fs::exists(dest));

    // Cancel keeps the partial file; the server honours Range for a later resume.
    fs::remove(dest + ".part", ec);
    std::atomic<bool> cancel{true};
    DownloadOptions stop;
    stop.cancel = &cancel;
    auto cancelled = download_file(s.url("/file"), dest, options(), stop);
    CHECK_FALSE(cancelled.ok);

    // Resume: 5000 bytes are already there, the rest is fetched with a Range request.
    fs::remove(dest + ".part", ec);
    { std::ofstream(dest + ".part", std::ios::binary) << content.substr(0, 5000); }
    DownloadOptions resume;
    resume.expected_size = content.size();
    auto resumed = download_file(s.url("/file"), dest, options(), resume);
    REQUIRE(resumed.ok);
    {
        std::ifstream in(dest, std::ios::binary);
        std::string got((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(got == content);
    }

    fs::remove_all(dir, ec);
}
