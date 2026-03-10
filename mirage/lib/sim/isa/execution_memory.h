#ifndef MIRAGE_SIM_ISA_EXECUTION_MEMORY_H_
#define MIRAGE_SIM_ISA_EXECUTION_MEMORY_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace mirage::sim::isa {

class ExecutionMemory {
 public:
  virtual ~ExecutionMemory() = default;

  virtual bool Load(std::uint64_t address,
                    std::span<std::byte> bytes) const = 0;
  virtual bool Store(std::uint64_t address,
                     std::span<const std::byte> bytes) = 0;

  virtual bool LoadU32(std::uint64_t address, std::uint32_t* value) const {
    if (value == nullptr) {
      return false;
    }
    std::array<std::byte, sizeof(std::uint32_t)> bytes{};
    if (!Load(address, std::span<std::byte>(bytes.data(), bytes.size()))) {
      return false;
    }
    std::memcpy(value, bytes.data(), sizeof(*value));
    return true;
  }

  virtual bool LoadU16(std::uint64_t address, std::uint16_t* value) const {
    if (value == nullptr) {
      return false;
    }
    std::array<std::byte, sizeof(std::uint16_t)> bytes{};
    if (!Load(address, std::span<std::byte>(bytes.data(), bytes.size()))) {
      return false;
    }
    std::memcpy(value, bytes.data(), sizeof(*value));
    return true;
  }

  virtual bool LoadU8(std::uint64_t address, std::uint8_t* value) const {
    if (value == nullptr) {
      return false;
    }
    std::array<std::byte, sizeof(std::uint8_t)> bytes{};
    if (!Load(address, std::span<std::byte>(bytes.data(), bytes.size()))) {
      return false;
    }
    std::memcpy(value, bytes.data(), sizeof(*value));
    return true;
  }

  virtual bool StoreU32(std::uint64_t address, std::uint32_t value) {
    std::array<std::byte, sizeof(std::uint32_t)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(value));
    return Store(address, std::span<const std::byte>(bytes.data(), bytes.size()));
  }

  virtual bool StoreU16(std::uint64_t address, std::uint16_t value) {
    std::array<std::byte, sizeof(std::uint16_t)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(value));
    return Store(address, std::span<const std::byte>(bytes.data(), bytes.size()));
  }

  virtual bool StoreU8(std::uint64_t address, std::uint8_t value) {
    std::array<std::byte, sizeof(std::uint8_t)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(value));
    return Store(address, std::span<const std::byte>(bytes.data(), bytes.size()));
  }
};

class LinearExecutionMemory final : public ExecutionMemory {
 public:
  explicit LinearExecutionMemory(std::size_t size_bytes,
                                 std::uint64_t base_address = 0)
      : base_address_(base_address), bytes_(size_bytes) {}

  bool Load(std::uint64_t address,
            std::span<std::byte> bytes) const override {
    if (!Contains(address, bytes.size())) {
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - base_address_);
    std::copy_n(bytes_.data() + offset, bytes.size(), bytes.data());
    return true;
  }

  bool Store(std::uint64_t address,
             std::span<const std::byte> bytes) override {
    if (!Contains(address, bytes.size())) {
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - base_address_);
    std::copy_n(bytes.data(), bytes.size(), bytes_.data() + offset);
    return true;
  }

  bool ReadU32(std::uint64_t address, std::uint32_t* value) const {
    return LoadU32(address, value);
  }

  bool WriteU32(std::uint64_t address, std::uint32_t value) {
    return StoreU32(address, value);
  }

  bool LoadU32(std::uint64_t address, std::uint32_t* value) const override {
    if (value == nullptr || !Contains(address, sizeof(*value))) {
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - base_address_);
    std::memcpy(value, bytes_.data() + offset, sizeof(*value));
    return true;
  }

  bool LoadU16(std::uint64_t address, std::uint16_t* value) const override {
    if (value == nullptr || !Contains(address, sizeof(*value))) {
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - base_address_);
    std::memcpy(value, bytes_.data() + offset, sizeof(*value));
    return true;
  }

  bool LoadU8(std::uint64_t address, std::uint8_t* value) const override {
    if (value == nullptr || !Contains(address, sizeof(*value))) {
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - base_address_);
    std::memcpy(value, bytes_.data() + offset, sizeof(*value));
    return true;
  }

  bool StoreU32(std::uint64_t address, std::uint32_t value) override {
    if (!Contains(address, sizeof(value))) {
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - base_address_);
    std::memcpy(bytes_.data() + offset, &value, sizeof(value));
    return true;
  }

  bool StoreU16(std::uint64_t address, std::uint16_t value) override {
    if (!Contains(address, sizeof(value))) {
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - base_address_);
    std::memcpy(bytes_.data() + offset, &value, sizeof(value));
    return true;
  }

  bool StoreU8(std::uint64_t address, std::uint8_t value) override {
    if (!Contains(address, sizeof(value))) {
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - base_address_);
    std::memcpy(bytes_.data() + offset, &value, sizeof(value));
    return true;
  }

 private:
  bool Contains(std::uint64_t address, std::size_t size_bytes) const {
    if (address < base_address_) {
      return false;
    }
    const std::uint64_t offset = address - base_address_;
    return offset <= bytes_.size() &&
           size_bytes <= bytes_.size() - static_cast<std::size_t>(offset);
  }

  std::uint64_t base_address_ = 0;
  std::vector<std::byte> bytes_;
};

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_EXECUTION_MEMORY_H_
