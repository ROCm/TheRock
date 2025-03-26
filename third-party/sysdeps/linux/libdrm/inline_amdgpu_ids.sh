#!/bin/bash
# Reads the data/amdgpu.ids file and replaces the amdgpu/amdgpu_asic_id.c file
# with a fully inlined table that does not need to consult a fixed place on
# the filesystem.

set -euo pipefail

SOURCE_DIR="${1:?Expected source directory}"

generate() {
  echo '
#include <ctype.h>
#include <stdint.h>
#include <string.h>
#include "amdgpu_drm.h"
#include "amdgpu_internal.h"
struct inline_amdgpu_id {
  uint32_t did;
  uint32_t rid;
  const char *name;
};
'
  # Output an inline table.
  echo 'static struct inline_amdgpu_id inline_amdgpu_ids[] = {'

  # Syntax: device_id,  revision_id,  product_name
  # Fields are separated by {comma} {tab}.
  while read -r line; do
    if [[ "$line" =~ ^# ]]; then
      # Skip comment lines.
      continue
    fi
    IFS=$',\t' read -ra parts <<< "$line"
    if [[ ${#parts[@]} != 3 ]]; then
      # Skip any lines that are not three fields. This also skips blanks and
      # the file version leading line.
      continue
    fi
    echo "  {0x${parts[0]}, 0x${parts[1]}, \"${parts[2]}\"},"
  done < "${SOURCE_DIR}/data/amdgpu.ids"
  echo '};'

  # And export a lookup function.
  echo '
void amdgpu_parse_asic_ids(struct amdgpu_device *dev) {
  const size_t count = sizeof inline_amdgpu_ids / sizeof inline_amdgpu_ids[0];
  for (size_t i = 0; i < count; ++i) {
    if (inline_amdgpu_ids[i].did == dev->info.asic_id &&
        inline_amdgpu_ids[i].rid == dev->info.pci_rev_id) {
      // Trim leading whitespace/tabs.
      const char *name = inline_amdgpu_ids[i].name;
      while (isblank(*name)) name++;
      if (strlen(name) == 0) continue;
      dev->marketing_name = strdup(name);
      break;
    }
  }
}
'
}

generate > "${SOURCE_DIR}/amdgpu/amdgpu_asic_id.c"
