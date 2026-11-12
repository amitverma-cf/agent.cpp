#include "secrets.hpp"

#include "cli_core.hpp"

#include <agent-cpp/hub.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <simdjson.h>

#if defined(_WIN32)
// clang-format off
#include <windows.h>
#include <wincred.h>
// clang-format on
#elif defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#endif

namespace agent::cli {

namespace fs = std::filesystem;

bool valid_secret_name(std::string_view name) {
    if (name.empty() || name.size() > 64) return false;
    return std::all_of(name.begin(), name.end(), [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '.' || c == '-'; });
}

namespace {

constexpr const char *kService = "agent.cpp";

Result<void> bad_name(std::string_view name) {
    return fail(ErrorCode::InvalidConfig, "secret name '" + std::string(name) + "' must be 1-64 characters from A-Z a-z 0-9 _ . -");
}

// ------------------------------------------------------------------------------------------------ Windows
#if defined(_WIN32)

std::wstring wide(std::string_view s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::wstring target_name(std::string_view name) { return wide(std::string(kService) + "/" + std::string(name)); }

bool wincred_available(void *) { return true; }

Result<void> wincred_set(void *, std::string_view name, std::string_view value) {
    if (!valid_secret_name(name)) return bad_name(name);
    if (value.size() > CRED_MAX_CREDENTIAL_BLOB_SIZE)
        return fail(ErrorCode::InvalidConfig,
                    "Windows Credential Manager holds at most " + std::to_string(CRED_MAX_CREDENTIAL_BLOB_SIZE) + " bytes per secret");
    const std::wstring target = target_name(name);
    CREDENTIALW cred{};
    cred.Type = CRED_TYPE_GENERIC;
    cred.TargetName = const_cast<LPWSTR>(target.c_str());
    cred.CredentialBlobSize = static_cast<DWORD>(value.size());
    cred.CredentialBlob = reinterpret_cast<LPBYTE>(const_cast<char *>(value.data()));
    cred.Persist = CRED_PERSIST_LOCAL_MACHINE;
    if (!CredWriteW(&cred, 0))
        return fail(ErrorCode::FilesystemError, "Credential Manager refused the secret (error " + std::to_string(GetLastError()) + ")");
    return ok();
}

Result<std::string> wincred_get(void *, std::string_view name) {
    if (!valid_secret_name(name)) return fail<std::string>(ErrorCode::InvalidConfig, "invalid secret name");
    const std::wstring target = target_name(name);
    PCREDENTIALW cred = nullptr;
    if (!CredReadW(target.c_str(), CRED_TYPE_GENERIC, 0, &cred))
        return fail<std::string>(ErrorCode::KeyNotFound, "no secret named '" + std::string(name) + "' in Windows Credential Manager");
    std::string value(reinterpret_cast<const char *>(cred->CredentialBlob), cred->CredentialBlobSize);
    SecureZeroMemory(cred->CredentialBlob, cred->CredentialBlobSize);
    CredFree(cred);
    return ok(std::move(value));
}

Result<void> wincred_remove(void *, std::string_view name) {
    if (!valid_secret_name(name)) return bad_name(name);
    const std::wstring target = target_name(name);
    if (!CredDeleteW(target.c_str(), CRED_TYPE_GENERIC, 0) && GetLastError() != ERROR_NOT_FOUND)
        return fail(ErrorCode::FilesystemError,
                    "Credential Manager could not delete the secret (error " + std::to_string(GetLastError()) + ")");
    return ok();
}

const SecretBackend kOsBackend = {"Windows Credential Manager", nullptr, wincred_available, wincred_set, wincred_get, wincred_remove};

// ------------------------------------------------------------------------------------------------ macOS (written from the API docs; not tested here)
#elif defined(__APPLE__)

struct CfString {
    CFStringRef ref;
    explicit CfString(std::string_view s)
        : ref(CFStringCreateWithBytes(nullptr, reinterpret_cast<const UInt8 *>(s.data()), static_cast<CFIndex>(s.size()),
                                      kCFStringEncodingUTF8, false)) {}
    ~CfString() {
        if (ref) CFRelease(ref);
    }
};

CFMutableDictionaryRef keychain_query(const CfString &account) {
    CFMutableDictionaryRef q = CFDictionaryCreateMutable(nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(q, kSecClass, kSecClassGenericPassword);
    static const CfString service(kService);
    CFDictionarySetValue(q, kSecAttrService, service.ref);
    CFDictionarySetValue(q, kSecAttrAccount, account.ref);
    return q;
}

bool keychain_available(void *) { return true; }

Result<void> keychain_set(void *, std::string_view name, std::string_view value) {
    if (!valid_secret_name(name)) return bad_name(name);
    const CfString account(name);
    CFMutableDictionaryRef q = keychain_query(account);
    SecItemDelete(q); // replace an existing one
    CFDataRef data = CFDataCreate(nullptr, reinterpret_cast<const UInt8 *>(value.data()), static_cast<CFIndex>(value.size()));
    CFDictionarySetValue(q, kSecValueData, data);
    const OSStatus st = SecItemAdd(q, nullptr);
    CFRelease(data);
    CFRelease(q);
    if (st != errSecSuccess) return fail(ErrorCode::FilesystemError, "Keychain refused the secret (status " + std::to_string(st) + ")");
    return ok();
}

Result<std::string> keychain_get(void *, std::string_view name) {
    if (!valid_secret_name(name)) return fail<std::string>(ErrorCode::InvalidConfig, "invalid secret name");
    const CfString account(name);
    CFMutableDictionaryRef q = keychain_query(account);
    CFDictionarySetValue(q, kSecReturnData, kCFBooleanTrue);
    CFDictionarySetValue(q, kSecMatchLimit, kSecMatchLimitOne);
    CFTypeRef found = nullptr;
    const OSStatus st = SecItemCopyMatching(q, &found);
    CFRelease(q);
    if (st != errSecSuccess || !found)
        return fail<std::string>(ErrorCode::KeyNotFound, "no secret named '" + std::string(name) + "' in the Keychain");
    CFDataRef data = static_cast<CFDataRef>(found);
    std::string value(reinterpret_cast<const char *>(CFDataGetBytePtr(data)), static_cast<size_t>(CFDataGetLength(data)));
    CFRelease(found);
    return ok(std::move(value));
}

Result<void> keychain_remove(void *, std::string_view name) {
    if (!valid_secret_name(name)) return bad_name(name);
    const CfString account(name);
    CFMutableDictionaryRef q = keychain_query(account);
    const OSStatus st = SecItemDelete(q);
    CFRelease(q);
    if (st != errSecSuccess && st != errSecItemNotFound)
        return fail(ErrorCode::FilesystemError, "Keychain could not delete the secret (status " + std::to_string(st) + ")");
    return ok();
}

const SecretBackend kOsBackend = {"macOS Keychain", nullptr, keychain_available, keychain_set, keychain_get, keychain_remove};

// ------------------------------------------------------------------------------------------------ Linux: Secret Service through `secret-tool` (written from its man page; not tested here)
#else

// The value goes through the tool's stdin, never through its command line. Names are validated, so the shell
// command built here contains nothing but a fixed program and a safe name.
bool secret_tool_available(void *) {
    if (std::system("command -v secret-tool >/dev/null 2>&1") != 0) return false;
    // A session bus is needed too: `search` fails fast without one.
    return std::system("secret-tool search service agent.cpp-probe >/dev/null 2>&1") == 0 ||
           std::getenv("DBUS_SESSION_BUS_ADDRESS") != nullptr;
}

Result<void> secret_tool_set(void *, std::string_view name, std::string_view value) {
    if (!valid_secret_name(name)) return bad_name(name);
    const std::string cmd = std::string("secret-tool store --label='agent.cpp ") + std::string(name) + "' service " + kService +
                            " account " + std::string(name) + " 2>/dev/null";
    FILE *p = popen(cmd.c_str(), "w");
    if (!p) return fail(ErrorCode::FilesystemError, "cannot start secret-tool");
    std::fwrite(value.data(), 1, value.size(), p);
    return pclose(p) == 0 ? ok()
                          : fail(ErrorCode::FilesystemError, "secret-tool could not store the secret (is a keyring running and unlocked?)");
}

Result<std::string> secret_tool_get(void *, std::string_view name) {
    if (!valid_secret_name(name)) return fail<std::string>(ErrorCode::InvalidConfig, "invalid secret name");
    const std::string cmd = std::string("secret-tool lookup service ") + kService + " account " + std::string(name) + " 2>/dev/null";
    FILE *p = popen(cmd.c_str(), "r");
    if (!p) return fail<std::string>(ErrorCode::FilesystemError, "cannot start secret-tool");
    std::string value;
    char buf[512];
    for (size_t n; (n = std::fread(buf, 1, sizeof(buf), p)) > 0;) value.append(buf, n);
    if (pclose(p) != 0 || value.empty())
        return fail<std::string>(ErrorCode::KeyNotFound, "no secret named '" + std::string(name) + "' in the keyring");
    return ok(std::move(value));
}

Result<void> secret_tool_remove(void *, std::string_view name) {
    if (!valid_secret_name(name)) return bad_name(name);
    const std::string cmd = std::string("secret-tool clear service ") + kService + " account " + std::string(name) + " 2>/dev/null";
    std::system(cmd.c_str()); // clearing something that is not there is fine
    return ok();
}

const SecretBackend kOsBackend = {
    "Secret Service (secret-tool)", nullptr, secret_tool_available, secret_tool_set, secret_tool_get, secret_tool_remove};
#endif

// ------------------------------------------------------------------------------------------------ encrypted file
//
// One file holds every secret, encrypted with AES-256-GCM under a key derived from a passphrase with scrypt.
// Layout: "AGSEC1" | salt[16] | iv[12] | tag[16] | ciphertext (a JSON object name -> value).

struct FileState {
    std::string path;
    std::string passphrase; // cached for the process; from $AGENT_SECRETS_PASSPHRASE or a prompt
    bool have_passphrase = false;
};

constexpr char kMagic[] = "AGSEC1";
constexpr size_t kMagicLen = 6, kSaltLen = 16, kIvLen = 12, kTagLen = 16;

bool derive_key(const std::string &pass, const unsigned char *salt, unsigned char *key) {
    return EVP_PBE_scrypt(pass.data(), pass.size(), salt, kSaltLen, 1 << 15, 8, 1, 128ULL * 1024 * 1024, key, 32) == 1;
}

Result<std::string> passphrase_of(FileState &st) {
    if (st.have_passphrase) return ok(st.passphrase);
    if (const char *env = std::getenv("AGENT_SECRETS_PASSPHRASE"); env && *env) {
        st.passphrase = env;
    } else if (stdin_is_tty()) {
        if (!read_hidden_line("Passphrase for the encrypted secrets file: ", st.passphrase) || st.passphrase.empty())
            return fail<std::string>(ErrorCode::InvalidConfig, "no passphrase given");
    } else {
        return fail<std::string>(
            ErrorCode::InvalidConfig,
            "this machine has no OS credential store, so secrets use an encrypted file: set AGENT_SECRETS_PASSPHRASE or run on a terminal");
    }
    st.have_passphrase = true;
    return ok(st.passphrase);
}

std::string encode_map(const std::map<std::string, std::string> &m) {
    std::string out = "{";
    bool first = true;
    for (const auto &[k, v] : m) {
        out += first ? "\"" : ",\"";
        first = false;
        out += json_escape(k) + "\":\"" + json_escape(v) + "\"";
    }
    return out + "}";
}

Result<std::map<std::string, std::string>> load_file_secrets(FileState &st) {
    using R = std::map<std::string, std::string>;
    std::error_code ec;
    if (!fs::is_regular_file(st.path, ec)) return ok(R{});
    std::ifstream in(st.path, std::ios::binary);
    std::string blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (blob.size() < kMagicLen + kSaltLen + kIvLen + kTagLen || blob.compare(0, kMagicLen, kMagic) != 0)
        return fail<R>(ErrorCode::ParseError, st.path + " is not an agent.cpp secrets file");
    auto pass = passphrase_of(st);
    if (!pass.ok) return fail<R>(pass.error.code, pass.error.message);
    const auto *p = reinterpret_cast<const unsigned char *>(blob.data()) + kMagicLen;
    unsigned char key[32];
    if (!derive_key(pass.value, p, key)) return fail<R>(ErrorCode::DecodeFailed, "key derivation failed");
    const unsigned char *iv = p + kSaltLen;
    const unsigned char *tag = iv + kIvLen;
    const unsigned char *cipher = tag + kTagLen;
    const int cipher_len = static_cast<int>(blob.size() - kMagicLen - kSaltLen - kIvLen - kTagLen);
    std::string plain(static_cast<size_t>(cipher_len), '\0');
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int n = 0, total = 0;
    bool good = EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, key, iv) == 1 &&
                EVP_DecryptUpdate(ctx, reinterpret_cast<unsigned char *>(plain.data()), &n, cipher, cipher_len) == 1;
    total = n;
    good = good && EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, static_cast<int>(kTagLen), const_cast<unsigned char *>(tag)) == 1 &&
           EVP_DecryptFinal_ex(ctx, reinterpret_cast<unsigned char *>(plain.data()) + total, &n) == 1;
    EVP_CIPHER_CTX_free(ctx);
    OPENSSL_cleanse(key, sizeof(key));
    if (!good) {
        st.have_passphrase = false; // let the next attempt ask again
        return fail<R>(ErrorCode::PermissionDenied, "wrong passphrase, or the secrets file is damaged");
    }
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object obj;
    if (parser.parse(plain).get(doc) || doc.get(obj)) return fail<R>(ErrorCode::ParseError, "the secrets file holds invalid data");
    R out;
    for (auto field : obj) {
        std::string_view v;
        if (!field.value.get(v)) out.emplace(std::string(field.key), std::string(v));
    }
    return ok(std::move(out));
}

Result<void> save_file_secrets(FileState &st, const std::map<std::string, std::string> &m) {
    auto pass = passphrase_of(st);
    if (!pass.ok) return fail(pass.error.code, pass.error.message);
    unsigned char salt[kSaltLen], iv[kIvLen], key[32], tag[kTagLen];
    if (RAND_bytes(salt, sizeof(salt)) != 1 || RAND_bytes(iv, sizeof(iv)) != 1)
        return fail(ErrorCode::DecodeFailed, "no random bytes available");
    if (!derive_key(pass.value, salt, key)) return fail(ErrorCode::DecodeFailed, "key derivation failed");
    const std::string plain = encode_map(m);
    std::string cipher(plain.size(), '\0');
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int n = 0, total = 0;
    bool good = EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, key, iv) == 1 &&
                EVP_EncryptUpdate(ctx, reinterpret_cast<unsigned char *>(cipher.data()), &n,
                                  reinterpret_cast<const unsigned char *>(plain.data()), static_cast<int>(plain.size())) == 1;
    total = n;
    good = good && EVP_EncryptFinal_ex(ctx, reinterpret_cast<unsigned char *>(cipher.data()) + total, &n) == 1 &&
           EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, static_cast<int>(kTagLen), tag) == 1;
    EVP_CIPHER_CTX_free(ctx);
    OPENSSL_cleanse(key, sizeof(key));
    if (!good) return fail(ErrorCode::DecodeFailed, "encryption failed");

    std::error_code ec;
    const fs::path p(st.path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    const std::string tmp = st.path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return fail(ErrorCode::FilesystemError, "cannot write " + tmp);
        out.write(kMagic, static_cast<std::streamsize>(kMagicLen));
        out.write(reinterpret_cast<const char *>(salt), sizeof(salt));
        out.write(reinterpret_cast<const char *>(iv), sizeof(iv));
        out.write(reinterpret_cast<const char *>(tag), sizeof(tag));
        out.write(cipher.data(), static_cast<std::streamsize>(cipher.size()));
        if (!out.flush()) return fail(ErrorCode::FilesystemError, "failed writing " + tmp);
    }
#if !defined(_WIN32)
    std::filesystem::permissions(tmp, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
#endif
    fs::rename(tmp, p, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "cannot replace " + st.path + ": " + ec.message());
    return ok();
}

bool file_available(void *) { return true; }

Result<void> file_set(void *state, std::string_view name, std::string_view value) {
    if (!valid_secret_name(name)) return bad_name(name);
    auto &st = *static_cast<FileState *>(state);
    auto all = load_file_secrets(st);
    if (!all.ok) return fail(all.error.code, all.error.message);
    all.value[std::string(name)] = std::string(value);
    return save_file_secrets(st, all.value);
}

Result<std::string> file_get(void *state, std::string_view name) {
    auto &st = *static_cast<FileState *>(state);
    auto all = load_file_secrets(st);
    if (!all.ok) return fail<std::string>(all.error.code, all.error.message);
    auto it = all.value.find(std::string(name));
    if (it == all.value.end())
        return fail<std::string>(ErrorCode::KeyNotFound, "no secret named '" + std::string(name) + "' in the encrypted file");
    return ok(it->second);
}

Result<void> file_remove(void *state, std::string_view name) {
    auto &st = *static_cast<FileState *>(state);
    auto all = load_file_secrets(st);
    if (!all.ok) return fail(all.error.code, all.error.message);
    if (all.value.erase(std::string(name)) == 0) return ok();
    return save_file_secrets(st, all.value);
}

} // namespace

SecretBackend encrypted_file_backend(const std::string &path) {
    // One state per file for the life of the process, so the passphrase is asked for once.
    static std::mutex mu;
    static std::map<std::string, std::unique_ptr<FileState>> states;
    std::lock_guard<std::mutex> lock(mu);
    auto &slot = states[path];
    if (!slot) {
        slot = std::make_unique<FileState>();
        slot->path = path;
    }
    return SecretBackend{"encrypted file (AES-256-GCM, scrypt)", slot.get(), file_available, file_set, file_get, file_remove};
}

const SecretBackend &default_secret_backend(const std::string &home_root) {
    static SecretBackend chosen = [&] {
        if (kOsBackend.available(kOsBackend.state)) return kOsBackend;
        return encrypted_file_backend((fs::path(home_root) / "secrets.enc").string());
    }();
    return chosen;
}

// ------------------------------------------------------------------------------------------------ index, access, placeholders

Result<std::vector<SecretEntry>> load_secret_index(const std::string &path) {
    using R = std::vector<SecretEntry>;
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return ok(R{});
    std::ifstream in(path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(text).get(doc)) return fail<R>(ErrorCode::ParseError, path + " is not valid JSON");
    simdjson::dom::array secrets;
    if (doc["secrets"].get(secrets)) return fail<R>(ErrorCode::ParseError, path + " has no \"secrets\" array");
    R out;
    for (simdjson::dom::element s : secrets) {
        SecretEntry e;
        std::string_view v;
        if (!s["name"].get(v)) e.name = std::string(v);
        simdjson::dom::array bindings;
        if (!s["bindings"].get(bindings))
            for (simdjson::dom::element b : bindings) {
                SecretBinding sb;
                if (!b["consumer"].get(v)) sb.consumer = std::string(v);
                if (!b["identity"].get(v)) sb.identity = std::string(v);
                e.bindings.push_back(std::move(sb));
            }
        if (!e.name.empty()) out.push_back(std::move(e));
    }
    return ok(std::move(out));
}

Result<void> save_secret_index(const std::string &path, const std::vector<SecretEntry> &entries) {
    std::string out = "{\n  \"secrets\": [";
    for (size_t i = 0; i < entries.size(); ++i) {
        out += i ? ",\n    {" : "\n    {";
        out += "\"name\": \"" + json_escape(entries[i].name) + "\", \"bindings\": [";
        for (size_t b = 0; b < entries[i].bindings.size(); ++b)
            out += std::string(b ? ", " : "") + "{\"consumer\": \"" + json_escape(entries[i].bindings[b].consumer) +
                   "\", \"identity\": \"" + json_escape(entries[i].bindings[b].identity) + "\"}";
        out += "]}";
    }
    out += entries.empty() ? "]\n}\n" : "\n  ]\n}\n";
    std::error_code ec;
    const fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return fail(ErrorCode::FilesystemError, "cannot write " + tmp);
        f << out;
        if (!f.flush()) return fail(ErrorCode::FilesystemError, "failed writing " + tmp);
    }
    fs::rename(tmp, p, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "cannot replace " + path + ": " + ec.message());
    return ok();
}

std::string consumer_identity(const std::vector<std::string> &parts) {
    std::string joined;
    for (const std::string &p : parts) {
        joined += std::to_string(p.size());
        joined += ':';
        joined += p;
    }
    return agent::hub::sha256_hex(joined);
}

SecretAccess check_secret_access(const std::vector<SecretEntry> &index, std::string_view name, std::string_view consumer,
                                 std::string_view identity) {
    const auto it = std::find_if(index.begin(), index.end(), [&](const SecretEntry &e) { return e.name == name; });
    if (it == index.end()) return SecretAccess::UnknownSecret;
    bool same_consumer = false;
    for (const SecretBinding &b : it->bindings) {
        if (b.consumer != consumer) continue;
        same_consumer = true;
        if (b.identity == identity) return SecretAccess::Allowed;
    }
    return same_consumer ? SecretAccess::IdentityChanged : SecretAccess::NotBound;
}

Result<std::string> resolve_secret_placeholders(std::string_view text, const SecretBackend &backend, const std::vector<SecretEntry> &index,
                                                std::string_view consumer, std::string_view identity, std::vector<std::string> *values) {
    constexpr std::string_view kOpen = "${secret:";
    std::string out;
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t at = text.find(kOpen, pos);
        if (at == std::string_view::npos) {
            out.append(text.substr(pos));
            break;
        }
        out.append(text.substr(pos, at - pos));
        const size_t close = text.find('}', at + kOpen.size());
        if (close == std::string_view::npos) return fail<std::string>(ErrorCode::ParseError, "unterminated ${secret:...} placeholder");
        const std::string_view name = text.substr(at + kOpen.size(), close - at - kOpen.size());
        switch (check_secret_access(index, name, consumer, identity)) {
        case SecretAccess::Allowed: break;
        case SecretAccess::UnknownSecret:
            return fail<std::string>(ErrorCode::KeyNotFound, "secret '" + std::string(name) +
                                                                 "' does not exist: add it with `agent secret set " + std::string(name) +
                                                                 "`");
        case SecretAccess::NotBound:
            return fail<std::string>(ErrorCode::PermissionDenied, "secret '" + std::string(name) + "' is not bound to " +
                                                                      std::string(consumer) + ": allow it with `agent secret bind " +
                                                                      std::string(name) + " " + std::string(consumer) + "`");
        case SecretAccess::IdentityChanged:
            return fail<std::string>(ErrorCode::PermissionDenied,
                                     std::string(consumer) + " now runs something different from what secret '" + std::string(name) +
                                         "' was approved for; approve it again with `agent secret bind " + std::string(name) + " " +
                                         std::string(consumer) + "`");
        }
        auto value = backend.get(backend.state, name);
        if (!value.ok) return fail<std::string>(value.error.code, value.error.message);
        out += value.value;
        if (values) values->push_back(value.value);
        pos = close + 1;
    }
    return ok(std::move(out));
}

Result<std::shared_ptr<SecretContext>> load_secret_context(const std::string &home_root, const std::string &index_path) {
    auto index = load_secret_index(index_path);
    if (!index.ok) return fail<std::shared_ptr<SecretContext>>(index.error.code, index.error.message);
    auto ctx = std::make_shared<SecretContext>();
    ctx->backend = default_secret_backend(home_root);
    ctx->index = std::move(index.value);
    return ok(std::move(ctx));
}

Result<std::string> cli_resolve_secrets(std::string_view text, const SecretRequest &who, std::vector<std::string> &secret_values,
                                        void *user_data) {
    const auto *ctx = static_cast<const SecretContext *>(user_data);
    if (!ctx) return fail<std::string>(ErrorCode::InvalidConfig, "no secrets are configured");
    const std::vector<std::string> parts(who.identity_parts.begin(), who.identity_parts.end());
    return resolve_secret_placeholders(text, ctx->backend, ctx->index, who.consumer, consumer_identity(parts), &secret_values);
}

std::string redact_secrets(std::string text, const std::vector<std::string> &values) {
    for (const std::string &v : values) {
        if (v.size() < 4) continue; // shorter values would mask ordinary text
        for (size_t at = text.find(v); at != std::string::npos; at = text.find(v, at + 3)) text.replace(at, v.size(), "***");
    }
    return text;
}

} // namespace agent::cli
