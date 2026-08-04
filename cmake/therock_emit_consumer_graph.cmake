# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_emit_consumer_graph(output_file)
#
# Serializes the THEROCK_ALL_SUBPROJECTS / THEROCK_CONSUMERS_OF_* global
# property registry (populated by therock_cmake_subproject_declare) to a
# JSON file at <output_file>.
#
# Must be called AFTER all therock_cmake_subproject_declare() calls have run
# (i.e. at the end of the top-level CMakeLists.txt).  The output is a
# configure-time side-effect (not a build target), analogous to
# compile_commands.json.
#
# JSON schema:
#   {
#     "<subproject-lowercase>": {
#       "consumers": ["<consumer-lowercase>", ...]
#     },
#     ...
#   }
#
# The graph carries only consumer (reverse-dependency) edges.  No build-stage,
# artifact, or other facts are duplicated here — each keeps its own single source
# of truth.  Test selection walks these consumer edges by hop distance
# (see test_tools/determine_rocm_test_dependencies.py).
#
# Lifecycle: this configure-time emit writes build/therock_consumer_graph.json,
# but that copy is a GENERATED artifact used to refresh the COMMITTED graph at
# test_tools/therock_consumer_graph.json.  The committed copy is what the per-PR
# change-detection job reads directly — no fetch, no configure on the hot path.
# A low-frequency drift-check job (.github/workflows/consumer_graph_drift.yml)
# regenerates the graph from a configure, normalizes it, and fails if it differs
# from the committed copy, prompting the contributor to regenerate and re-commit.
# The committed graph is generated-only and must NEVER be hand-edited.
function(therock_emit_consumer_graph output_file)
  get_property(_all GLOBAL PROPERTY THEROCK_ALL_SUBPROJECTS)

  set(_json "{\n")
  set(_first_proj TRUE)

  foreach(_proj IN LISTS _all)
    get_property(_consumers GLOBAL PROPERTY "THEROCK_CONSUMERS_OF_${_proj}")
    if(_consumers)
      list(REMOVE_DUPLICATES _consumers)
    endif()

    if(NOT _first_proj)
      string(APPEND _json ",\n")
    endif()
    set(_first_proj FALSE)

    string(TOLOWER "${_proj}" _key)

    # Build JSON array of consumers.
    set(_arr "")
    set(_first_c TRUE)
    foreach(_c IN LISTS _consumers)
      string(TOLOWER "${_c}" _cv)
      if(NOT _first_c)
        string(APPEND _arr ", ")
      endif()
      set(_first_c FALSE)
      string(APPEND _arr "\"${_cv}\"")
    endforeach()

    string(APPEND _json "  \"${_key}\": {\n")
    string(APPEND _json "    \"consumers\": [${_arr}]\n")
    string(APPEND _json "  }")
  endforeach()

  string(APPEND _json "\n}\n")
  file(WRITE "${output_file}" "${_json}")
  message(STATUS "Wrote consumer graph to ${output_file}")
endfunction()
