# Vendored third-party code

## `cmake_parser` (cmake-parser 0.9.2)

- **Source:** PyPI `cmake-parser==0.9.2` sdist (`src/cmake_parser/`).
- **License:** Apache-2.0 — see `cmake_parser/LICENSE`.
- **Why vendored:** it is a sub-1.0 dependency on a CI-critical path (the consumer-graph drift check runs in
  `unit_tests.yml` on every PR). Vendoring pins the exact bytes so a yanked or changed upstream release cannot
  break every PR. See issue #7782.

### How to update

```bash
python -m pip download "cmake-parser==<version>" --no-deps --no-binary :all: -d /tmp/cmp
tar -xzf /tmp/cmp/cmake_parser-<version>.tar.gz -C /tmp/cmp
cp -r /tmp/cmp/cmake_parser-<version>/src/cmake_parser build_tools/consumer_graph/_vendor/cmake_parser
cp /tmp/cmp/cmake_parser-<version>/LICENSES/Apache-2.0.txt build_tools/consumer_graph/_vendor/cmake_parser/LICENSE
```

### Import wiring

The `cmake_consumer_graph` package imports `cmake_parser` as a top-level module. It is made importable from this
vendored copy (rather than a pip install) by prepending `build_tools/consumer_graph/_vendor` to `sys.path` in the
package `__init__` (and in `conftest.py` for the tests).
