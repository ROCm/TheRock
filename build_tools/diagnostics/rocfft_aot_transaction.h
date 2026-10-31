// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

// Experiment only: a private, regenerable AOT intermediate. Runtime caches and
// caller-supplied paths must keep their existing transaction/durability policy.
#include "sqlite3.h"
#include <stdexcept>
#include <string>

namespace rocfft_build_profile {
inline bool private_aot_batch_requested(const std::string &setting,
                                        const std::string &supplied_path,
                                        bool default_cache_exists) {
  if (setting.empty() || setting == "0")
    return false;
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
