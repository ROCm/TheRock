// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <rocjpeg/rocjpeg.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kCases[] = {
    "rocjpeg.stream.invalid",
    "rocjpeg.stream.mug_400",
    "rocjpeg.stream.mug_420",
    "rocjpeg.stream.mug_422",
};

int fail(std::string_view message) {
  std::cerr << "FAIL: " << message << '\n';
  return 1;
}

int parse_bytes(const std::vector<std::uint8_t> &bytes,
                RocJpegStatus expected) {
  RocJpegStreamHandle stream = nullptr;
  if (rocJpegStreamCreate(&stream) != ROCJPEG_STATUS_SUCCESS ||
      stream == nullptr) {
    return fail("rocJpegStreamCreate failed");
  }
  const RocJpegStatus status =
      rocJpegStreamParse(bytes.data(), bytes.size(), stream);
  const RocJpegStatus destroy_status = rocJpegStreamDestroy(stream);
  if (status != expected) {
    return fail("rocJpegStreamParse returned an unexpected status");
  }
  if (destroy_status != ROCJPEG_STATUS_SUCCESS) {
    return fail("rocJpegStreamDestroy failed");
  }
  return 0;
}

int run_invalid_case() {
  const std::vector<std::vector<std::uint8_t>> inputs = {
      {0xFF, 0x00},
      {0xFF, 0xD8, 0xFF, 0xDD, 0x00, 0x03},
      {0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x01, 0x04},
  };
  for (const auto &input : inputs) {
    if (parse_bytes(input, ROCJPEG_STATUS_BAD_JPEG) != 0) {
      return 1;
    }
  }
  std::cout << "PASS rocjpeg.stream.invalid\n";
  return 0;
}

int run_file_case(std::string_view name,
                  const std::filesystem::path &data_root) {
  const std::string suffix(name.substr(name.rfind('_') + 1));
  const auto input = data_root / "images" / ("mug_" + suffix + ".jpg");
  std::ifstream stream(input, std::ios::binary);
  if (!stream) {
    return fail("missing input " + input.string());
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(stream)),
                                  std::istreambuf_iterator<char>());
  if (bytes.empty()) {
    return fail("empty JPEG input");
  }
  if (parse_bytes(bytes, ROCJPEG_STATUS_SUCCESS) != 0) {
    return 1;
  }
  std::cout << "PASS " << name << '\n';
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--list") {
    for (const auto name : kCases) {
      std::cout << name << '\n';
    }
    return 0;
  }
  if (argc != 4 || std::string_view(argv[1]) != "--case") {
    return fail("usage: rocjpeg_host_tsan --list | --case NAME DATA_ROOT");
  }

  const std::string_view requested(argv[2]);
  if (requested == "rocjpeg.stream.invalid") {
    return run_invalid_case();
  }
  for (const auto name : kCases) {
    if (name == requested) {
      return run_file_case(name, argv[3]);
    }
  }
  return fail("unknown case");
}
