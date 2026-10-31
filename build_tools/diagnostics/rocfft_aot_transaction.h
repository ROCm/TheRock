// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

// Experiment only: a private, regenerable AOT intermediate. Runtime caches and
// caller-supplied paths must keep their existing transaction/durability policy.
#include "sqlite3.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace rocfft_build_profile {
struct AotCacheSelection {
  std::string path;
  bool owned;
  bool existed;
};

// Accidental-reuse guard for a controller-created directory, not a security
// boundary against an adversarial process changing files concurrently.
inline AotCacheSelection select_private_aot_cache(const std::string &supplied,
                                                  const std::string &owned_path,
                                                  const std::string &owner_id) {
  namespace fs = std::filesystem;
  if (owned_path.empty()) {
    if (!owner_id.empty())
      throw std::invalid_argument("AOT owner ID requires an owned cache path");
    const auto path = supplied.empty() ? "rocfft_temp_cache.db" : supplied;
    // Preserve the frozen legacy probe, including for caller-supplied paths.
    return {path, false, fs::exists("rocfft_temp_cache.db")};
  }
  if (!supplied.empty() || owner_id.size() != 32 ||
      !std::all_of(owner_id.begin(), owner_id.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    throw std::invalid_argument(
        "Owned AOT cache requires default argv and owner ID");
  const fs::path path(owned_path), directory = path.parent_path();
  const auto marker = directory / ".rocfft-aot-owner";
  if (!path.is_absolute() || path.filename() != "rocfft_temp_cache.db" ||
      fs::is_symlink(directory) || !fs::is_directory(directory) ||
      fs::canonical(directory) != directory.lexically_normal() ||
      fs::is_symlink(marker) || !fs::is_regular_file(marker) ||
      fs::hard_link_count(marker) != 1 || fs::is_symlink(path) ||
      (fs::exists(path) &&
       (!fs::is_regular_file(path) || fs::hard_link_count(path) != 1)))
    throw std::invalid_argument("Invalid owned AOT cache path or marker");
  for (const char *suffix : {"-wal", "-shm"}) {
    const fs::path sidecar(path.string() + suffix);
    if (fs::is_symlink(sidecar) ||
        (fs::exists(sidecar) &&
         (!fs::is_regular_file(sidecar) || fs::hard_link_count(sidecar) != 1)))
      throw std::invalid_argument("Invalid owned AOT cache sidecar");
  }
  std::ifstream input(marker, std::ios::binary);
  std::array<char, 34> text{};
  input.read(text.data(), text.size());
  if (input.gcount() != 33 || std::string(text.data(), 33) != owner_id + "\n")
    throw std::invalid_argument(
        "Owned AOT cache marker does not match owner ID");
  return {path.string(), true, fs::exists(path)};
}

inline bool private_aot_batch_requested(const std::string &setting,
                                        const std::string &supplied_path,
                                        bool default_cache_exists,
                                        bool owned = false) {
  if (setting.empty() || setting == "0")
    return false;
  if (setting == "all") {
    if (!owned)
      throw std::invalid_argument(
          "All-helper batching requires an owned cache");
    return true;
  }
  if (setting != "1" || !supplied_path.empty())
    throw std::invalid_argument("AOT batching requires value 1 and the "
                                "helper-owned default temp cache");
  // Later architecture helpers reuse the growing private DB with original
  // autocommit behavior. Only the first fresh helper is the experimental arm.
  return !default_cache_exists;
}

inline void private_aot_transaction(sqlite3 *db, bool begin) {
  if (!db || (sqlite3_get_autocommit(db) != 0) != begin)
    throw std::runtime_error("Unexpected private AOT transaction state");
  // Do not change journal_mode, synchronous, busy timeout or checkpoint policy.
  // A failure aborts the helper; SQLite rolls back an unfinished private batch
  // when its connection is closed. No incomplete output is published as
  // success.
  const int result = sqlite3_exec(db, begin ? "BEGIN IMMEDIATE" : "COMMIT",
                                  nullptr, nullptr, nullptr);
  if (result != SQLITE_OK)
    throw std::runtime_error(std::string("Private AOT transaction failed: ") +
                             sqlite3_errmsg(db));
}
} // namespace rocfft_build_profile
