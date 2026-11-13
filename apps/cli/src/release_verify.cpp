#include "release_verify.hpp"

#include <memory>
#include <openssl/evp.h>
#include <vector>

namespace agent::cli {

bool base64_to_bytes(std::string_view text, std::string &out) {
    out.clear();
    unsigned buffer = 0;
    int bits = 0;
    for (const char c : text) {
        int v;
        if (c >= 'A' && c <= 'Z') v = c - 'A';
        else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
        else if (c >= '0' && c <= '9') v = c - '0' + 52;
        else if (c == '+') v = 62;
        else if (c == '/') v = 63;
        else if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        else return false;
        buffer = (buffer << 6) | static_cast<unsigned>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += static_cast<char>((buffer >> bits) & 0xFF);
        }
    }
    return !out.empty();
}

bool verify_ed25519(std::string_view message, std::string_view signature_b64, std::string_view public_key_b64) {
    std::string signature, key;
    if (!base64_to_bytes(signature_b64, signature) || !base64_to_bytes(public_key_b64, key) || key.size() != 32 || signature.size() != 64)
        return false;
    EVP_PKEY *pkey =
        EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, nullptr, reinterpret_cast<const unsigned char *>(key.data()), key.size());
    if (!pkey) return false;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    bool valid = false;
    if (ctx && EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, pkey) == 1)
        valid = EVP_DigestVerify(ctx, reinterpret_cast<const unsigned char *>(signature.data()), signature.size(),
                                 reinterpret_cast<const unsigned char *>(message.data()), message.size()) == 1;
    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(pkey);
    return valid;
}

} // namespace agent::cli
