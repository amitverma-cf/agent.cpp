#include "archive.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------- inflate
// A compact canonical-Huffman decoder in the style of zlib's "puff".

constexpr int kMaxBits = 15;
constexpr int kMaxLCodes = 286;
constexpr int kMaxDCodes = 30;
constexpr int kFixLCodes = 288;

struct Huffman {
    std::array<uint16_t, kMaxBits + 1> count{};
    std::array<uint16_t, kFixLCodes> symbol{};
};

class Inflater {
  public:
    Inflater(std::span<const uint8_t> in, size_t max_output) : in_(in), max_(max_output) {}

    Result<std::string> run() {
        bool last = false;
        while (!last) {
            int v = 0;
            if (!bits(1, v)) return truncated();
            last = v != 0;
            int type = 0;
            if (!bits(2, type)) return truncated();
            int rc = 0;
            if (type == 0) rc = stored();
            else if (type == 1) rc = fixed();
            else if (type == 2) rc = dynamic();
            else return fail<std::string>(ErrorCode::ParseError, "deflate: invalid block type");
            if (rc == kTruncated) return truncated();
            if (rc == kTooBig) return fail<std::string>(ErrorCode::ParseError, "deflate: output is larger than the limit");
            if (rc != 0) return fail<std::string>(ErrorCode::ParseError, "deflate: corrupt data");
        }
        return ok(std::move(out_));
    }

    size_t consumed() const { return pos_; }

  private:
    static constexpr int kTruncated = -100;
    static constexpr int kTooBig = -101;

    static Result<std::string> truncated() { return fail<std::string>(ErrorCode::ParseError, "deflate: the data ends early"); }

    bool bits(int need, int &value) {
        long val = bitbuf_;
        while (bitcnt_ < need) {
            if (pos_ >= in_.size()) return false;
            val |= static_cast<long>(in_[pos_++]) << bitcnt_;
            bitcnt_ += 8;
        }
        bitbuf_ = static_cast<int>(val >> need);
        bitcnt_ -= need;
        value = static_cast<int>(val & ((1L << need) - 1));
        return true;
    }

    int stored() {
        bitbuf_ = 0;
        bitcnt_ = 0;
        if (pos_ + 4 > in_.size()) return kTruncated;
        const unsigned len = in_[pos_] | (in_[pos_ + 1] << 8);
        const unsigned nlen = in_[pos_ + 2] | (in_[pos_ + 3] << 8);
        pos_ += 4;
        if (len != (~nlen & 0xffffu)) return -1;
        if (pos_ + len > in_.size()) return kTruncated;
        if (out_.size() + len > max_) return kTooBig;
        out_.append(reinterpret_cast<const char *>(in_.data() + pos_), len);
        pos_ += len;
        return 0;
    }

    // Returns 0 for a complete code, > 0 for an incomplete one, < 0 for an over-subscribed set.
    static int construct(Huffman &h, const uint16_t *length, int n) {
        h.count.fill(0);
        for (int s = 0; s < n; ++s) ++h.count[length[s]];
        if (h.count[0] == n) return 0;
        int left = 1;
        for (int len = 1; len <= kMaxBits; ++len) {
            left <<= 1;
            left -= h.count[len];
            if (left < 0) return left;
        }
        std::array<uint16_t, kMaxBits + 1> offs{};
        for (int len = 1; len < kMaxBits; ++len) offs[len + 1] = static_cast<uint16_t>(offs[len] + h.count[len]);
        for (int s = 0; s < n; ++s)
            if (length[s] != 0) h.symbol[offs[length[s]]++] = static_cast<uint16_t>(s);
        return left;
    }

    int decode(const Huffman &h) {
        int code = 0, first = 0, index = 0;
        for (int len = 1; len <= kMaxBits; ++len) {
            int bit = 0;
            if (!bits(1, bit)) return kTruncated;
            code |= bit;
            const int count = h.count[len];
            if (code - count < first) return h.symbol[index + (code - first)];
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        return -1;
    }

    int codes(const Huffman &lencode, const Huffman &distcode) {
        static constexpr uint16_t lens[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
        static constexpr uint16_t lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
        static constexpr uint16_t dists[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
        static constexpr uint16_t dext[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
        for (;;) {
            int symbol = decode(lencode);
            if (symbol < 0) return symbol;
            if (symbol < 256) {
                if (out_.size() + 1 > max_) return kTooBig;
                out_.push_back(static_cast<char>(symbol));
            } else if (symbol == 256) {
                return 0;
            } else {
                symbol -= 257;
                if (symbol >= 29) return -1;
                int extra = 0;
                if (!bits(lext[symbol], extra)) return kTruncated;
                const size_t len = static_cast<size_t>(lens[symbol] + extra);
                const int dsym = decode(distcode);
                if (dsym < 0) return dsym;
                if (dsym >= 30) return -1;
                if (!bits(dext[dsym], extra)) return kTruncated;
                const size_t dist = static_cast<size_t>(dists[dsym] + extra);
                if (dist > out_.size()) return -1;
                if (out_.size() + len > max_) return kTooBig;
                size_t from = out_.size() - dist;
                for (size_t i = 0; i < len; ++i) out_.push_back(out_[from + i]); // may overlap: byte by byte
            }
        }
    }

    int fixed() {
        static Huffman lencode, distcode;
        static bool built = false;
        if (!built) {
            uint16_t lengths[kFixLCodes];
            int s = 0;
            for (; s < 144; ++s) lengths[s] = 8;
            for (; s < 256; ++s) lengths[s] = 9;
            for (; s < 280; ++s) lengths[s] = 7;
            for (; s < kFixLCodes; ++s) lengths[s] = 8;
            construct(lencode, lengths, kFixLCodes);
            for (s = 0; s < kMaxDCodes; ++s) lengths[s] = 5;
            construct(distcode, lengths, kMaxDCodes);
            built = true;
        }
        return codes(lencode, distcode);
    }

    int dynamic() {
        static constexpr uint16_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
        int nlen = 0, ndist = 0, ncode = 0;
        if (!bits(5, nlen) || !bits(5, ndist) || !bits(4, ncode)) return kTruncated;
        nlen += 257;
        ndist += 1;
        ncode += 4;
        if (nlen > kMaxLCodes || ndist > kMaxDCodes) return -1;
        uint16_t lengths[kMaxLCodes + kMaxDCodes] = {};
        int index = 0;
        for (; index < ncode; ++index) {
            int v = 0;
            if (!bits(3, v)) return kTruncated;
            lengths[order[index]] = static_cast<uint16_t>(v);
        }
        for (; index < 19; ++index) lengths[order[index]] = 0;
        Huffman lencode, distcode;
        if (construct(lencode, lengths, 19) != 0) return -1;
        index = 0;
        while (index < nlen + ndist) {
            int symbol = decode(lencode);
            if (symbol < 0) return symbol;
            if (symbol < 16) {
                lengths[index++] = static_cast<uint16_t>(symbol);
            } else {
                uint16_t len = 0;
                int rep = 0;
                if (symbol == 16) {
                    if (index == 0) return -1;
                    len = lengths[index - 1];
                    int v = 0;
                    if (!bits(2, v)) return kTruncated;
                    rep = 3 + v;
                } else if (symbol == 17) {
                    int v = 0;
                    if (!bits(3, v)) return kTruncated;
                    rep = 3 + v;
                } else {
                    int v = 0;
                    if (!bits(7, v)) return kTruncated;
                    rep = 11 + v;
                }
                if (index + rep > nlen + ndist) return -1;
                while (rep--) lengths[index++] = len;
            }
        }
        if (lengths[256] == 0) return -1;
        int err = construct(lencode, lengths, nlen);
        if (err < 0 || (err > 0 && nlen - lencode.count[0] != 1)) return -1;
        err = construct(distcode, lengths + nlen, ndist);
        if (err < 0 || (err > 0 && ndist - distcode.count[0] != 1)) return -1;
        return codes(lencode, distcode);
    }

    std::span<const uint8_t> in_;
    size_t max_;
    size_t pos_ = 0;
    int bitbuf_ = 0;
    int bitcnt_ = 0;
    std::string out_;
};

uint32_t le32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24); }
uint16_t le16(const uint8_t *p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

// ---------------------------------------------------------------- safe destination paths

// Turns an archive entry name into a path under `dest`, or "" to skip it. Sets `bad` for an unsafe name.
std::string entry_path(std::string name, int strip, bool &bad) {
    bad = false;
    std::replace(name.begin(), name.end(), '\\', '/');
    if (name.empty()) return {};
    if (name[0] == '/' || (name.size() > 1 && name[1] == ':')) {
        bad = true;
        return {};
    }
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= name.size()) {
        size_t end = name.find('/', start);
        if (end == std::string::npos) end = name.size();
        std::string part = name.substr(start, end - start);
        start = end + 1;
        if (part.empty() || part == ".") continue;
        if (part == "..") {
            bad = true;
            return {};
        }
        parts.push_back(std::move(part));
    }
    if (static_cast<int>(parts.size()) <= strip) return {};
    std::string out;
    for (size_t i = static_cast<size_t>(strip); i < parts.size(); ++i) {
        if (!out.empty()) out += '/';
        out += parts[i];
    }
    return out;
}

Result<void> write_entry(const std::string &dest_dir, const std::string &rel, std::string_view data, bool executable) {
    std::error_code ec;
    const fs::path target = fs::path(dest_dir) / fs::path(rel);
    fs::create_directories(target.parent_path(), ec);
    if (ec) return fail(ErrorCode::FilesystemError, "cannot create " + target.parent_path().string() + ": " + ec.message());
    {
        std::ofstream out(target, std::ios::binary | std::ios::trunc);
        if (!out) return fail(ErrorCode::FilesystemError, "cannot write " + target.string());
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
        if (!out) return fail(ErrorCode::FilesystemError, "cannot write " + target.string());
    }
    if (executable) {
        fs::permissions(target, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::add, ec);
    }
    return ok();
}

} // namespace

Result<std::string> inflate_raw(std::span<const uint8_t> in, size_t max_output) {
    Inflater inflater(in, max_output);
    return inflater.run();
}

Result<std::string> gunzip(std::span<const uint8_t> in, size_t max_output) {
    if (in.size() < 18 || in[0] != 0x1f || in[1] != 0x8b || in[2] != 8) return fail<std::string>(ErrorCode::ParseError, "gzip: not a gzip stream");
    const uint8_t flags = in[3];
    size_t pos = 10;
    if (flags & 4) { // FEXTRA
        if (pos + 2 > in.size()) return fail<std::string>(ErrorCode::ParseError, "gzip: truncated header");
        pos += 2 + le16(in.data() + pos);
    }
    for (uint8_t bit : {uint8_t{8}, uint8_t{16}}) { // FNAME, FCOMMENT: zero-terminated
        if (!(flags & bit)) continue;
        while (pos < in.size() && in[pos] != 0) ++pos;
        ++pos;
    }
    if (flags & 2) pos += 2; // FHCRC
    if (pos >= in.size()) return fail<std::string>(ErrorCode::ParseError, "gzip: truncated header");
    return inflate_raw(in.subspan(pos), max_output);
}

Result<void> extract_zip_bytes(std::span<const uint8_t> zip, const std::string &dest_dir, int strip_components) {
    // The end-of-central-directory record is within the last 64 KiB + 22 bytes.
    if (zip.size() < 22) return fail(ErrorCode::ParseError, "zip: too small");
    size_t eocd = std::string::npos;
    const size_t lowest = zip.size() > 65557 ? zip.size() - 65557 : 0;
    for (size_t i = zip.size() - 22 + 1; i-- > lowest;) {
        if (le32(zip.data() + i) == 0x06054b50u) {
            eocd = i;
            break;
        }
    }
    if (eocd == std::string::npos) return fail(ErrorCode::ParseError, "zip: no central directory");
    const size_t entries = le16(zip.data() + eocd + 10);
    size_t pos = le32(zip.data() + eocd + 16);
    for (size_t n = 0; n < entries; ++n) {
        if (pos + 46 > zip.size() || le32(zip.data() + pos) != 0x02014b50u) return fail(ErrorCode::ParseError, "zip: bad central directory");
        const uint8_t *c = zip.data() + pos;
        const uint16_t made_by = le16(c + 4);
        const uint16_t method = le16(c + 10);
        const uint32_t comp = le32(c + 20), size = le32(c + 24);
        const uint16_t name_len = le16(c + 28), extra_len = le16(c + 30), comment_len = le16(c + 32);
        const uint32_t external = le32(c + 38), local = le32(c + 42);
        if (pos + 46 + name_len > zip.size()) return fail(ErrorCode::ParseError, "zip: bad entry name");
        const std::string name(reinterpret_cast<const char *>(c + 46), name_len);
        pos += 46u + name_len + extra_len + comment_len;

        bool bad = false;
        const std::string rel = entry_path(name, strip_components, bad);
        if (bad) return fail(ErrorCode::PermissionDenied, "zip: refusing the entry '" + name + "' (it would leave the destination)");
        if (rel.empty() || name.back() == '/') continue;
        if (comp == 0xffffffffu || size == 0xffffffffu) return fail(ErrorCode::InvalidConfig, "zip: zip64 entries are not supported");

        if (static_cast<size_t>(local) + 30 > zip.size() || le32(zip.data() + local) != 0x04034b50u) return fail(ErrorCode::ParseError, "zip: bad local header");
        const uint8_t *l = zip.data() + local;
        const size_t data_at = static_cast<size_t>(local) + 30 + le16(l + 26) + le16(l + 28);
        if (data_at + comp > zip.size()) return fail(ErrorCode::ParseError, "zip: entry data is out of range");
        std::string content;
        if (method == 0) {
            content.assign(reinterpret_cast<const char *>(zip.data() + data_at), comp);
        } else if (method == 8) {
            auto inflated = inflate_raw(zip.subspan(data_at, comp), size_t{1} << 31);
            if (!inflated.ok) return fail(inflated.error.code, "zip: " + name + ": " + inflated.error.message);
            content = std::move(inflated.value);
        } else {
            return fail(ErrorCode::InvalidConfig, "zip: " + name + " uses an unsupported compression method");
        }
        if (content.size() != size) return fail(ErrorCode::ParseError, "zip: " + name + " has the wrong size");
        const bool exec = (made_by >> 8) == 3 && ((external >> 16) & 0111) != 0; // created on Unix: mode in the high 16 bits
        if (auto w = write_entry(dest_dir, rel, content, exec); !w.ok) return w;
    }
    return ok();
}

Result<void> extract_tar_bytes(std::span<const uint8_t> tar, const std::string &dest_dir, int strip_components) {
    auto octal = [](const uint8_t *p, size_t n) {
        uint64_t v = 0;
        for (size_t i = 0; i < n && p[i] >= '0' && p[i] <= '7'; ++i) v = v * 8 + (p[i] - '0');
        return v;
    };
    auto field = [](const uint8_t *p, size_t n) {
        size_t len = 0;
        while (len < n && p[len] != 0) ++len;
        return std::string(reinterpret_cast<const char *>(p), len);
    };
    std::string long_name; // from a GNU 'L' or pax 'x' record, applies to the next entry
    size_t pos = 0;
    while (pos + 512 <= tar.size()) {
        const uint8_t *h = tar.data() + pos;
        if (std::all_of(h, h + 512, [](uint8_t b) { return b == 0; })) break;
        const uint64_t size = octal(h + 124, 12);
        const char type = static_cast<char>(h[156]);
        const size_t data_at = pos + 512;
        if (data_at + size > tar.size()) return fail(ErrorCode::ParseError, "tar: entry data is out of range");
        const std::string_view data(reinterpret_cast<const char *>(tar.data() + data_at), static_cast<size_t>(size));
        pos = data_at + static_cast<size_t>((size + 511) / 512 * 512);

        if (type == 'L') { // GNU long name
            long_name = std::string(data.substr(0, data.find('\0')));
            continue;
        }
        if (type == 'x') { // pax header: "<len> path=<name>\n"
            size_t at = 0;
            while (at < data.size()) {
                const size_t sp = data.find(' ', at);
                if (sp == std::string_view::npos) break;
                const size_t len = static_cast<size_t>(std::strtoul(std::string(data.substr(at, sp - at)).c_str(), nullptr, 10));
                if (len == 0 || at + len > data.size()) break;
                std::string_view rec = data.substr(sp + 1, at + len - sp - 2);
                if (rec.starts_with("path=")) long_name = std::string(rec.substr(5));
                at += len;
            }
            continue;
        }
        if (type == 'g') continue;

        const bool has_long_name = !long_name.empty();
        std::string name = has_long_name ? long_name : field(h, 100);
        long_name.clear();
        if (!has_long_name && field(h + 257, 5) == "ustar") { // ustar splits a long name into prefix + name
            const std::string prefix = field(h + 345, 155);
            if (!prefix.empty()) name = prefix + "/" + name;
        }

        bool bad = false;
        const std::string rel = entry_path(name, strip_components, bad);
        if (bad) return fail(ErrorCode::PermissionDenied, "tar: refusing the entry '" + name + "' (it would leave the destination)");
        if (rel.empty()) continue;
        if (type == '5') {
            std::error_code ec;
            fs::create_directories(fs::path(dest_dir) / fs::path(rel), ec);
            continue;
        }
        if (type != '0' && type != '\0') continue; // links and special files are not needed for the runtimes
        const bool exec = (octal(h + 100, 8) & 0111) != 0;
        if (auto w = write_entry(dest_dir, rel, data, exec); !w.ok) return w;
    }
    return ok();
}

Result<void> extract_archive(const std::string &archive_path, const std::string &dest_dir, int strip_components) {
    std::ifstream in(archive_path, std::ios::binary);
    if (!in) return fail(ErrorCode::FilesystemError, "cannot read " + archive_path);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string lower = [&] {
        std::string s = archive_path;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }();
    if (lower.ends_with(".zip")) return extract_zip_bytes(bytes, dest_dir, strip_components);
    if (lower.ends_with(".tar.gz") || lower.ends_with(".tgz")) {
        auto tar = gunzip(bytes);
        if (!tar.ok) return fail(tar.error.code, archive_path + ": " + tar.error.message);
        return extract_tar_bytes(std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(tar.value.data()), tar.value.size()), dest_dir, strip_components);
    }
    return fail(ErrorCode::InvalidConfig, "unsupported archive type: " + archive_path);
}

} // namespace agent::cli
