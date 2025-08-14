# Prefetch LibElf to nullify the shady embedded find modules.
# Maddeningly, the find module calls itself "LibElf" (camel case) but sets
# "LIBELF_FOUND" (uppercase).
find_package(LibElf CONFIG REQUIRED)
set(LIBELF_FOUND ON)
