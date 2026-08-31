# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_emit_consumer_graph(output_file)
#
# Serializes the consumer graph (from the THEROCK_ALL_SUBPROJECTS /
# THEROCK_DIRECT_CONSUMERS_OF_* global properties) to JSON at <output_file>. Call
# at the end of the top-level CMakeLists.txt, after all subprojects are declared.
#
# Schema (reverse-dependency edges only):
#   { "<subproject>": { "consumers": ["<consumer>", ...] }, ... }
#
# The emit writes build/therock_consumer_graph.json, which refreshes the committed
# copy at test_tools/therock_consumer_graph.json. The committed copy is read
# directly by CI (no configure on the hot path) and kept honest by the drift check
# in .github/workflows/test_consumer_graph_drift.yml. It is generated-only; never
# hand-edit it.
function(therock_emit_consumer_graph output_file)
  get_property(_all GLOBAL PROPERTY THEROCK_ALL_SUBPROJECTS)

  set(_json "{\n")
  set(_first_proj TRUE)

  foreach(_proj IN LISTS _all)
    get_property(_consumers GLOBAL PROPERTY "THEROCK_DIRECT_CONSUMERS_OF_${_proj}")
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
