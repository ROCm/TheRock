# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(WIN32)
    set(EMBED_USE RC CACHE STRING "Use RC or CArrays to embed data files")
    set_property(CACHE EMBED_USE PROPERTY STRINGS "RC;CArrays")
else()
    if(BUILD_SHARED_LIBS)
        set(EMBED_USE LD CACHE STRING "Use LD or CArrays to embed data files")
    else()
        set(EMBED_USE CArrays CACHE STRING "Use LD or CArrays to embed data files")
    endif()
    set_property(CACHE EMBED_USE PROPERTY STRINGS "LD;CArrays")
endif()

if(EMBED_USE STREQUAL "LD")
    find_program(EMBED_LD ld REQUIRED)
    find_program(EMBED_OBJCOPY objcopy REQUIRED)
endif()

function(embed_wrap_string)
    set(options)
    set(oneValueArgs VARIABLE AT_COLUMN)
    set(multiValueArgs)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    string(LENGTH ${${PARSE_VARIABLE}} string_length)
    math(EXPR offset "0")

    while(string_length GREATER 0)

        if(string_length GREATER ${PARSE_AT_COLUMN})
            math(EXPR length "${PARSE_AT_COLUMN}")
        else()
            math(EXPR length "${string_length}")
        endif()

        string(SUBSTRING ${${PARSE_VARIABLE}} ${offset} ${length} line)
        set(lines "${lines}\n${line}")

        math(EXPR string_length "${string_length} - ${length}")
        math(EXPR offset "${offset} + ${length}")
    endwhile()

    set(${PARSE_VARIABLE} "${lines}" PARENT_SCOPE)
endfunction()

function(generate_embed_source EMBED_NAME EMBED_DIR BASE_DIRECTORY)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SYMBOLS FILES)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(RESOURCE_ID 100)

    list(LENGTH PARSE_SYMBOLS SYMBOLS_LEN)
    list(LENGTH PARSE_FILES FILES_LEN)
    if(NOT ${SYMBOLS_LEN} EQUAL ${FILES_LEN})
        message(FATAL_ERROR "Symbols and objects dont match: ${SYMBOLS_LEN} != ${FILES_LEN}")
    endif()
    math(EXPR LEN "${SYMBOLS_LEN} - 1")

    foreach(idx RANGE ${LEN})
        list(GET PARSE_SYMBOLS ${idx} SYMBOL)
        list(GET PARSE_FILES ${idx} FILE)
        file(RELATIVE_PATH BASE_NAME "${BASE_DIRECTORY}" ${FILE})
        if(EMBED_USE STREQUAL "RC")
            string(TOUPPER "${SYMBOL}" SYMBOL)
            string(APPEND FILE_IDS "#define IDR_${SYMBOL} ${RESOURCE_ID}\n")
            file(TO_NATIVE_PATH "${FILE}" NATIVE_FILE)
            string(REPLACE "\\" "\\\\" NATIVE_FILE "${NATIVE_FILE}")
            string(APPEND RC_FILE_MAPPING "IDR_${SYMBOL} TEXTFILE \"${NATIVE_FILE}\"\n")
            string(APPEND INIT_KERNELS "\n        {\"${BASE_NAME}\", resource::read(IDR_${SYMBOL})},")
            math(EXPR RESOURCE_ID "${RESOURCE_ID} + 1" OUTPUT_FORMAT DECIMAL)
        else()
            set(START_SYMBOL "_binary_${SYMBOL}_start")
            set(LENGTH_SYMBOL "_binary_${SYMBOL}_length")
            if(EMBED_USE STREQUAL "LD")
                string(APPEND EXTERNS "
extern const char ${START_SYMBOL}[];
extern const size_t _binary_${SYMBOL}_size;
const auto ${LENGTH_SYMBOL} = reinterpret_cast<size_t>(&_binary_${SYMBOL}_size);
")
            else()
                string(APPEND EXTERNS "
extern const char ${START_SYMBOL}[];
extern const size_t ${LENGTH_SYMBOL};
")
            endif()
            string(APPEND INIT_KERNELS "
        { \"${BASE_NAME}\", { ${START_SYMBOL}, ${LENGTH_SYMBOL}} },")
        endif()
    endforeach()
    if(EMBED_USE STREQUAL "RC")
       file(WRITE "${EMBED_DIR}/include/resource.h" "
#define TEXTFILE 256

${FILE_IDS}
")
        file(WRITE "${EMBED_DIR}/resource.rc" "
#include \"resource.h\"

${RC_FILE_MAPPING}
")
        set(EXTERNS "
#include <Windows.h>
#include \"resource.h\"

namespace resource {
std::string_view read(int id)
{
    HMODULE handle = GetModuleHandle(nullptr);
    HRSRC rc = FindResource(handle, MAKEINTRESOURCE(id), MAKEINTRESOURCE(TEXTFILE));
    HGLOBAL data = LoadResource(handle, rc);
    return {static_cast<const char*>(LockResource(data)), SizeofResource(handle, rc)};
}
}
")
        set(EMBED_FILES ${EMBED_DIR}/include/resource.h ${EMBED_DIR}/resource.rc)
    endif()
    file(WRITE "${EMBED_DIR}/include/${EMBED_NAME}.hpp" "
#include <string_view>
#include <unordered_map>
#include <utility>
std::unordered_map<std::string_view, std::string_view> ${EMBED_NAME}();
")

    file(WRITE "${EMBED_DIR}/${EMBED_NAME}.cpp" "
#include <${EMBED_NAME}.hpp>
${EXTERNS}
std::unordered_map<std::string_view, std::string_view> ${EMBED_NAME}()
{
    static std::unordered_map<std::string_view, std::string_view> result = {${INIT_KERNELS}
    };
    return result;
}
")
    list(APPEND EMBED_FILES ${EMBED_DIR}/${EMBED_NAME}.cpp ${EMBED_DIR}/include/${EMBED_NAME}.hpp)
    set(EMBED_FILES ${EMBED_FILES} PARENT_SCOPE)
endfunction()

function(embed_file FILE BASE_DIRECTORY)
    message(STATUS "    ${FILE}")
    file(RELATIVE_PATH REL_FILE "${BASE_DIRECTORY}" ${FILE})
    string(MAKE_C_IDENTIFIER "${REL_FILE}" OUTPUT_SYMBOL)
    get_filename_component(OUTPUT_FILE_DIR "${REL_FILE}" DIRECTORY)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE_DIR}")
    if(EMBED_USE STREQUAL "LD")
        set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REL_FILE}.o")
        add_custom_command(
            OUTPUT "${OUTPUT_FILE}"
            COMMAND ${EMBED_LD} -r -o "${OUTPUT_FILE}" -z noexecstack --format=binary "${REL_FILE}"
            COMMAND ${EMBED_OBJCOPY} --rename-section .data=.rodata,alloc,load,readonly,data,contents "${OUTPUT_FILE}"
            WORKING_DIRECTORY "${BASE_DIRECTORY}"
            DEPENDS "${FILE}"
            VERBATIM)
        set(OUTPUT_FILE ${OUTPUT_FILE} PARENT_SCOPE)
    elseif(EMBED_USE STREQUAL "CArrays")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${FILE})
        set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REL_FILE}.cpp")
        # reads source file contents as hex string
        file(READ ${FILE} HEX_STRING HEX)
        # wraps the hex string into multiple lines
        embed_wrap_string(VARIABLE HEX_STRING AT_COLUMN 80)
        # adds '0x' prefix and comma suffix before and after every byte respectively
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " ARRAY_VALUES ${HEX_STRING})
        # removes trailing comma
        string(REGEX REPLACE ", $" "" ARRAY_VALUES ${ARRAY_VALUES})
        file(WRITE "${OUTPUT_FILE}" "
#include <cstddef>
extern const char _binary_${OUTPUT_SYMBOL}_start[] = { ${ARRAY_VALUES} };
extern const size_t _binary_${OUTPUT_SYMBOL}_length = sizeof(_binary_${OUTPUT_SYMBOL}_start);
")
        set(OUTPUT_FILE ${OUTPUT_FILE} PARENT_SCOPE)
    endif()
    set(OUTPUT_SYMBOL ${OUTPUT_SYMBOL} PARENT_SCOPE)
endfunction()

function(add_embed_library EMBED_NAME)
    set(options)
    set(oneValueArgs RELATIVE)
    set(multiValueArgs)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(EMBED_DIR ${CMAKE_CURRENT_BINARY_DIR}/embed/${EMBED_NAME})
    file(MAKE_DIRECTORY ${EMBED_DIR})
    message(STATUS "Embedding kernel files:")
    foreach(FILE ${PARSE_UNPARSED_ARGUMENTS})
        embed_file(${FILE} ${PARSE_RELATIVE})
        list(APPEND OUTPUT_FILES ${OUTPUT_FILE})
        list(APPEND SYMBOLS ${OUTPUT_SYMBOL})
    endforeach()
    message(STATUS "Generating embedding library '${EMBED_NAME}'")
    generate_embed_source(${EMBED_NAME} ${EMBED_DIR} "${PARSE_RELATIVE}" SYMBOLS ${SYMBOLS} FILES ${PARSE_UNPARSED_ARGUMENTS})
    set(INTERNAL_EMBED_LIB embed_lib_${EMBED_NAME})
    if(EMBED_USE STREQUAL "LD")
        add_library(${INTERNAL_EMBED_LIB} STATIC ${EMBED_FILES} ${OUTPUT_FILES})
    else()
        add_library(${INTERNAL_EMBED_LIB} OBJECT ${EMBED_FILES})
    endif()
    if(EMBED_USE STREQUAL "CArrays")
        target_sources(${INTERNAL_EMBED_LIB} PRIVATE ${OUTPUT_FILES})
    endif()
    target_include_directories(${INTERNAL_EMBED_LIB} PRIVATE "${EMBED_DIR}/include")
    target_compile_options(${INTERNAL_EMBED_LIB} PRIVATE -Wno-reserved-identifier -Wno-extern-initializer -Wno-missing-variable-declarations)
    set_target_properties(${INTERNAL_EMBED_LIB} PROPERTIES POSITION_INDEPENDENT_CODE On)
    add_library(${EMBED_NAME} INTERFACE)
    if(EMBED_USE STREQUAL "RC")
        target_link_libraries(${EMBED_NAME} INTERFACE $<TARGET_OBJECTS:${INTERNAL_EMBED_LIB}>)
    elseif(EMBED_USE STREQUAL "LD")
        target_link_libraries(${EMBED_NAME} INTERFACE ${INTERNAL_EMBED_LIB})
    else()
        target_sources(${EMBED_NAME} INTERFACE $<TARGET_OBJECTS:${INTERNAL_EMBED_LIB}>)
    endif()
    target_include_directories(${EMBED_NAME} INTERFACE 
	    $<BUILD_INTERFACE:${EMBED_DIR}/include>
	        $<INSTALL_INTERFACE:include/ck>)
endfunction()

