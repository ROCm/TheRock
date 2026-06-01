// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rtc/hip.hpp>
#include <rtc/compile_kernel.hpp>
#ifdef HIPRTC_FOR_CODEGEN_TESTS
#include <hip/hiprtc.h>
#include <rtc/manage_ptr.hpp>
#endif
#include <rtc/tmp_dir.hpp>
#include <algorithm>
#include <cassert>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace rtc {

bool EndsWith(const std::string& value, const std::string& suffix)
{
    if(suffix.size() > value.size())
        return false;
    else
        return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

std::vector<std::string> SplitString(const std::string& s, char delim)
{
    std::vector<std::string> elems;
    std::stringstream ss(s + delim);
    std::string item;
    while(std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}

template <class T>
T generic_read_file(const std::string& filename, size_t offset = 0, size_t nbytes = 0)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if(nbytes == 0)
    {
        // if there is a non-zero offset and nbytes is not set,
        // calculate size of remaining bytes to read
        nbytes = is.tellg();
        if(offset > nbytes)
            throw std::runtime_error("offset is larger than file size");
        nbytes -= offset;
    }
    if(nbytes < 1)
        throw std::runtime_error("Invalid size for: " + filename);
    is.seekg(offset, std::ios::beg);

    T buffer(nbytes, 0);
    if(not is.read(&buffer[0], nbytes))
        throw std::runtime_error("Error reading file: " + filename);
    return buffer;
}

std::vector<char> read_buffer(const std::string& filename, size_t offset = 0, size_t nbytes = 0)
{
    return generic_read_file<std::vector<char>>(filename, offset, nbytes);
}

std::string read_string(const std::string& filename)
{
    return generic_read_file<std::string>(filename);
}

void write_buffer(const std::string& filename, const char* buffer, std::size_t size)
{
    std::ofstream os(filename);
    os.write(buffer, size);
}
void write_buffer(const std::string& filename, const std::vector<char>& buffer)
{
    write_buffer(filename, buffer.data(), buffer.size());
}
void write_string(const std::string& filename, const std::string_view& buffer)
{
    write_buffer(filename, buffer.data(), buffer.size());
}

std::string compiler() { return "/opt/rocm/llvm/bin/clang++ -x hip --cuda-device-only"; }
// TODO: undo after extracting the codeobj
// std::string compiler() { return "/opt/rocm/llvm/bin/clang++ -x hip"; }

kernel clang_compile_kernel(const std::vector<src_file>& srcs, compile_options options)
{
    assert(not srcs.empty());
    tmp_dir td{"compile"};
    options.flags += " -I. -O3";
    options.flags += " -std=c++20";
    options.flags += " --offload-arch=" + get_device_name();
    std::string out;

    for(const auto& src : srcs)
    {
        fs::path full_path   = td.path / src.path;
        fs::path parent_path = full_path.parent_path();
        fs::create_directories(parent_path);
        write_string(full_path.string(), src.content);
        if(src.path.extension().string() == ".cpp")
        {
            options.flags += " -c " + src.path.filename().string();
            if(out.empty())
                out = src.path.stem().string() + ".o";
        }
    }

    options.flags += " -o " + out;
    td.execute(compiler() + options.flags);

    auto out_path = td.path / out;
    if(not fs::exists(out_path))
        throw std::runtime_error("Output file missing: " + out);

    auto obj = read_buffer(out_path.string());

    std::ofstream ofh("obj.o", std::ios::binary);
    for(auto i : obj)
        ofh << i;
    ofh.close();
    // int s = std::system(("/usr/bin/cp " + out_path.string() + " codeobj.bin").c_str());
    // assert(s == 0);
    return kernel{obj.data(), options.kernel_name};
}

#ifdef HIPRTC_FOR_CODEGEN_TESTS

std::string hiprtc_error(hiprtcResult err, const std::string& msg)
{
    return "hiprtc: " + (hiprtcGetErrorString(err) + (": " + msg));
}

void hiprtc_check_error(hiprtcResult err, const std::string& msg = "")
{
    if(err != HIPRTC_SUCCESS)
        throw std::runtime_error(hiprtc_error(err, msg));
}

struct hiprtc_src_file
{
    hiprtc_src_file() = default;
    hiprtc_src_file(const src_file& s) : path(s.path.string()), content(s.content) {}
    std::string path;
    std::string content;
};

void hiprtc_program_destroy(hiprtcProgram prog) { hiprtcDestroyProgram(&prog); }
using hiprtc_program_ptr = RTC_MANAGE_PTR(hiprtcProgram, hiprtc_program_destroy);

template <class... Ts>
hiprtc_program_ptr hiprtc_program_create(Ts... xs)
{
    hiprtcProgram prog = nullptr;
    auto result        = hiprtcCreateProgram(&prog, xs...);
    hiprtc_program_ptr p{prog};
    hiprtc_check_error(result, "Create program failed.");
    return p;
}

struct hiprtc_program
{
    struct string_array
    {
        std::deque<std::string> strings{};
        std::vector<const char*> c_strs{};

        string_array() {}
        string_array(const string_array&) = delete;

        std::size_t size() const { return strings.size(); }

        const char** data() { return c_strs.data(); }

        void push_back(std::string s)
        {
            strings.push_back(std::move(s));
            c_strs.push_back(strings.back().c_str());
        }
    };

    hiprtc_program_ptr prog = nullptr;
    string_array headers{};
    string_array include_names{};
    std::string cpp_src  = "";
    std::string cpp_name = "";

    hiprtc_program(const std::string& src, const std::string& name = "main.cpp")
        : cpp_src(src), cpp_name(name)
    {
        create_program();
    }

    hiprtc_program(std::vector<src_file> srcs)
    {
        for(auto&& src : srcs)
        {
            if(EndsWith(src.path, ".cpp"))
            {
                cpp_src  = std::move(src.content);
                cpp_name = std::move(src.path);
            }
            else
            {
                headers.push_back(std::move(src.content));
                include_names.push_back(std::move(src.path));
            }
        }
        create_program();
    }

    void create_program()
    {
        assert(not cpp_src.empty());
        assert(not cpp_name.empty());
        assert(headers.size() == include_names.size());
        prog = hiprtc_program_create(cpp_src.c_str(),
                                     cpp_name.c_str(),
                                     headers.size(),
                                     headers.data(),
                                     include_names.data());
    }

    void compile(const std::vector<std::string>& options, bool quiet = false) const
    {
        std::vector<const char*> c_options;
        std::transform(options.begin(),
                       options.end(),
                       std::back_inserter(c_options),
                       [](const std::string& s) { return s.c_str(); });
        auto result   = hiprtcCompileProgram(prog.get(), c_options.size(), c_options.data());
        auto prog_log = log();
        if(not prog_log.empty() and not quiet)
        {
            std::cerr << prog_log << std::endl;
        }
        if(result != HIPRTC_SUCCESS)
            throw std::runtime_error("Compilation failed.");
    }

    std::string log() const
    {
        std::size_t n = 0;
        hiprtc_check_error(hiprtcGetProgramLogSize(prog.get(), &n));
        if(n == 0)
            return {};
        std::string buffer(n, '\0');
        hiprtc_check_error(hiprtcGetProgramLog(prog.get(), buffer.data()));
        assert(buffer.back() != 0);
        return buffer;
    }

    std::vector<char> get_code_obj() const
    {
        std::size_t n = 0;
        hiprtc_check_error(hiprtcGetCodeSize(prog.get(), &n));
        std::vector<char> buffer(n);
        hiprtc_check_error(hiprtcGetCode(prog.get(), buffer.data()));
        return buffer;
    }
};

std::vector<std::vector<char>> compile_hip_src_with_hiprtc(const std::vector<src_file>& srcs,
                                                           const compile_options& options)
{
    hiprtc_program prog(srcs);
    auto flags = SplitString(options.flags, ' ');
    prog.compile(flags);
    return {prog.get_code_obj()};
}

static kernel hiprtc_compile_kernel(const std::vector<src_file>& srcs, compile_options options)
{
    options.flags += " -I. -O3";
    options.flags += " -std=c++20";
    options.flags += " -DCK_CODE_GEN_RTC";
    options.flags += " --offload-arch=" + get_device_name();
    auto cos = compile_hip_src_with_hiprtc(srcs, options);
    if(cos.size() != 1)
        std::runtime_error("No code object");
    auto& obj = cos.front();
    return kernel{obj.data(), options.kernel_name};
}

#endif

kernel compile_kernel(const std::vector<src_file>& srcs, compile_options options)
{
#ifdef HIPRTC_FOR_CODEGEN_TESTS
    return hiprtc_compile_kernel(srcs, options);
#else
    return clang_compile_kernel(srcs, options);
#endif
}

} // namespace rtc
