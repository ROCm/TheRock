# Pinned rocSHMEM target-generation fixture

`pinned-source.zip` contains the exact four upstream files identified by
`source-index.json`, including their license headers. ZIP member names match the
index. The tests verify each original SHA-256 before applying the normal source
patch. Keeping the raw bytes in an archive prevents Git line-ending conversion
and source-formatting hooks from changing the pinned evidence.

The fixture generates actual patched target CMake with CMake/Ninja and compares
all compile commands and the two native-link commands. Its surrounding stubs do
not compile ROCm sources or validate runtime GPU behavior.
