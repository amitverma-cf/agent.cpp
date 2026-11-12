#pragma once

// Unpacking for the private runtimes (bun, uv): DEFLATE, .zip and .tar.gz, with no third-party library.
// Entries that would land outside the destination (absolute paths, "..", drive letters) are refused.

#include <agent-cpp/agent.hpp>
#include <cstdint>
#include <span>
#include <string>

namespace agent::cli {

// RFC 1951 raw DEFLATE. `max_output` bounds the result (a decompression bomb fails instead of using the memory).
Result<std::string> inflate_raw(std::span<const uint8_t> in, size_t max_output = size_t{1} << 31);

// RFC 1952 gzip (single member).
Result<std::string> gunzip(std::span<const uint8_t> in, size_t max_output = size_t{1} << 31);

// strip_components removes that many leading path parts from every entry (like tar --strip-components);
// entries with fewer parts are skipped. Executable bits are kept on POSIX.
Result<void> extract_zip_bytes(std::span<const uint8_t> zip, const std::string &dest_dir, int strip_components = 0);
Result<void> extract_tar_bytes(std::span<const uint8_t> tar, const std::string &dest_dir, int strip_components = 0);

// Reads the file and picks the format from the extension (.zip, .tar.gz, .tgz).
Result<void> extract_archive(const std::string &archive_path, const std::string &dest_dir, int strip_components = 0);

} // namespace agent::cli
