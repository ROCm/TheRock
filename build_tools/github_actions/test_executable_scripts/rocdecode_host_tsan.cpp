// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <rocdecode/roc_bitstream_reader.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

struct TestCase {
  std::string_view name;
  std::string_view file;
  rocDecVideoCodec codec;
};

constexpr TestCase kCases[] = {
    {"rocdecode.bitstream.av1", "video/AMD_driving_virtual_20-AV1.ivf",
     rocDecVideoCodec_AV1},
    {"rocdecode.bitstream.avc", "video/AMD_driving_virtual_20-H264.264",
     rocDecVideoCodec_AVC},
    {"rocdecode.bitstream.hevc", "video/AMD_driving_virtual_20-H265.265",
     rocDecVideoCodec_HEVC},
    {"rocdecode.bitstream.vp9", "video/AMD_driving_virtual_20-VP9.ivf",
     rocDecVideoCodec_VP9},
};

int fail(std::string_view message) {
  std::cerr << "FAIL: " << message << '\n';
  return 1;
}

int run_case(const TestCase &test, const std::filesystem::path &data_root) {
  const auto input = data_root / test.file;
  if (!std::filesystem::is_regular_file(input)) {
    return fail("missing input " + input.string());
  }

  RocdecBitstreamReader reader = nullptr;
  if (rocDecCreateBitstreamReader(&reader, input.c_str()) != ROCDEC_SUCCESS ||
      reader == nullptr) {
    return fail("rocDecCreateBitstreamReader failed");
  }

  auto destroy = [&reader]() {
    if (reader != nullptr) {
      rocDecDestroyBitstreamReader(reader);
      reader = nullptr;
    }
  };

  rocDecVideoCodec codec = rocDecVideoCodec_NumCodecs;
  if (rocDecGetBitstreamCodecType(reader, &codec) != ROCDEC_SUCCESS ||
      codec != test.codec) {
    destroy();
    return fail("unexpected bitstream codec");
  }

  int bit_depth = 0;
  if (rocDecGetBitstreamBitDepth(reader, &bit_depth) != ROCDEC_SUCCESS ||
      bit_depth <= 0) {
    destroy();
    return fail("invalid bitstream bit depth");
  }

  std::size_t packet_count = 0;
  for (;;) {
    std::uint8_t *data = nullptr;
    int size = 0;
    std::int64_t pts = 0;
    if (rocDecGetBitstreamPicData(reader, &data, &size, &pts) !=
        ROCDEC_SUCCESS) {
      destroy();
      return fail("rocDecGetBitstreamPicData failed");
    }
    if (size == 0) {
      break;
    }
    if (size < 0 || data == nullptr) {
      destroy();
      return fail("bitstream reader returned invalid picture data");
    }
    if (++packet_count > 10000) {
      destroy();
      return fail("bitstream reader did not reach end of stream");
    }
  }

  if (rocDecDestroyBitstreamReader(reader) != ROCDEC_SUCCESS) {
    reader = nullptr;
    return fail("rocDecDestroyBitstreamReader failed");
  }
  reader = nullptr;
  if (packet_count == 0) {
    return fail("bitstream contained no picture data");
  }

  std::cout << "PASS " << test.name << '\n';
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--list") {
    for (const auto &test : kCases) {
      std::cout << test.name << '\n';
    }
    return 0;
  }
  if (argc != 4 || std::string_view(argv[1]) != "--case") {
    return fail("usage: rocdecode_host_tsan --list | --case NAME DATA_ROOT");
  }

  const std::string_view requested(argv[2]);
  for (const auto &test : kCases) {
    if (test.name == requested) {
      return run_case(test, argv[3]);
    }
  }
  return fail("unknown case");
}
