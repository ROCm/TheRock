#include "lib/sim/single_gpu_simulator.h"

#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "lib/sim/isa/gfx950_binary_decoder.h"
#include "lib/sim/isa/gfx950_interpreter.h"

namespace mirage::sim {
namespace {

class SimulatorExecutionMemory final : public isa::ExecutionMemory {
 public:
  explicit SimulatorExecutionMemory(SingleGpuSimulator* simulator)
      : simulator_(simulator) {}

  bool Load(std::uint64_t address, std::span<std::byte> bytes) const override {
    return simulator_ != nullptr && simulator_->ReadMemory(address, bytes);
  }

  bool Store(std::uint64_t address,
             std::span<const std::byte> bytes) override {
    return simulator_ != nullptr && simulator_->WriteMemory(address, bytes);
  }

  bool LoadU32(std::uint64_t address, std::uint32_t* value) const override {
    if (simulator_ == nullptr || value == nullptr) {
      return false;
    }
    const auto loaded = simulator_->ReadObject<std::uint32_t>(address);
    if (!loaded.has_value()) {
      return false;
    }
    *value = *loaded;
    return true;
  }

  bool LoadU16(std::uint64_t address, std::uint16_t* value) const override {
    if (simulator_ == nullptr || value == nullptr) {
      return false;
    }
    const auto loaded = simulator_->ReadObject<std::uint16_t>(address);
    if (!loaded.has_value()) {
      return false;
    }
    *value = *loaded;
    return true;
  }

  bool LoadU8(std::uint64_t address, std::uint8_t* value) const override {
    if (simulator_ == nullptr || value == nullptr) {
      return false;
    }
    const auto loaded = simulator_->ReadObject<std::uint8_t>(address);
    if (!loaded.has_value()) {
      return false;
    }
    *value = *loaded;
    return true;
  }

  bool StoreU32(std::uint64_t address, std::uint32_t value) override {
    return simulator_ != nullptr && simulator_->WriteObject(address, value);
  }

  bool StoreU16(std::uint64_t address, std::uint16_t value) override {
    return simulator_ != nullptr && simulator_->WriteObject(address, value);
  }

  bool StoreU8(std::uint64_t address, std::uint8_t value) override {
    return simulator_ != nullptr && simulator_->WriteObject(address, value);
  }

 private:
  SingleGpuSimulator* simulator_ = nullptr;
};

}  // namespace

SingleGpuSimulator::SingleGpuSimulator(gpu::GpuProperties properties)
    : device_(std::move(properties)) {}

memory::AllocationHandle SingleGpuSimulator::AllocateMemory(
    memory::MemoryRegionKind kind,
    std::uint64_t size_bytes,
    std::uint64_t alignment) {
  if (size_bytes > std::numeric_limits<std::size_t>::max()) {
    return {};
  }

  const memory::GpuVaRange range = va_space_.Reserve(size_bytes, alignment);
  AllocationRecord record;
  record.handle.allocation_id = next_allocation_id_++;
  record.handle.kind = kind;
  record.handle.size_bytes = size_bytes;
  record.handle.mapped_va = range.base_va;
  record.bytes.resize(static_cast<std::size_t>(size_bytes));

  const auto [it, inserted] = allocations_.emplace(range.base_va, std::move(record));
  if (!inserted) {
    return {};
  }
  return it->second.handle;
}

bool SingleGpuSimulator::WriteMemory(std::uint64_t va,
                                     std::span<const std::byte> data) {
  if (data.empty()) {
    return true;
  }

  AllocationRecord* record = FindAllocation(va, data.size());
  if (record == nullptr) {
    return false;
  }

  const std::size_t offset =
      static_cast<std::size_t>(va - record->handle.mapped_va);
  std::memcpy(record->bytes.data() + offset, data.data(), data.size());
  ++record->write_version;
  return true;
}

bool SingleGpuSimulator::ReadMemory(std::uint64_t va,
                                    std::span<std::byte> data) const {
  if (data.empty()) {
    return true;
  }

  const AllocationRecord* record = FindAllocation(va, data.size());
  if (record == nullptr) {
    return false;
  }

  const std::size_t offset =
      static_cast<std::size_t>(va - record->handle.mapped_va);
  std::memcpy(data.data(), record->bytes.data() + offset, data.size());
  return true;
}

queue::QueueId SingleGpuSimulator::CreateComputeQueue(
    std::uint64_t ring_size_bytes) {
  return device_.CreateComputeQueue(ring_size_bytes);
}

std::optional<queue::QueueState> SingleGpuSimulator::GetQueue(
    queue::QueueId queue_id) const {
  return device_.QueryQueue(queue_id);
}

SingleGpuSimulator::DecodeCacheStats SingleGpuSimulator::GetDecodeCacheStats()
    const {
  DecodeCacheStats stats;
  stats.hits = decode_cache_hits_;
  stats.misses = decode_cache_misses_;
  stats.entries = decoded_program_cache_.size();
  return stats;
}

exec::CompletionRecord SingleGpuSimulator::Submit(
    queue::QueueId queue_id,
    const exec::SyntheticDispatchPacket& packet) {
  const auto queue_state = device_.QueryQueue(queue_id);
  if (!queue_state.has_value()) {
    return {};
  }
  if (queue_state->descriptor.type != queue::QueueType::kCompute) {
    return {};
  }

  const std::uint64_t dispatch_id = next_dispatch_id_++;
  const std::uint64_t next_write_ptr = queue_state->doorbell.write_ptr + 1;
  if (!device_.RingDoorbell(queue_id, next_write_ptr)) {
    return {};
  }

  exec::CompletionRecord completion = ExecuteDispatch(dispatch_id, packet);
  device_.RetireTo(queue_id, next_write_ptr);
  return completion;
}

SingleGpuSimulator::AllocationRecord* SingleGpuSimulator::FindAllocation(
    std::uint64_t va,
    std::size_t size_bytes) {
  auto it = allocations_.upper_bound(va);
  if (it == allocations_.begin()) {
    return nullptr;
  }
  --it;

  const std::uint64_t base_va = it->second.handle.mapped_va;
  if (va < base_va) {
    return nullptr;
  }

  const std::uint64_t offset = va - base_va;
  if (offset > it->second.bytes.size()) {
    return nullptr;
  }
  if (size_bytes > it->second.bytes.size() - static_cast<std::size_t>(offset)) {
    return nullptr;
  }
  return &it->second;
}

const SingleGpuSimulator::AllocationRecord* SingleGpuSimulator::FindAllocation(
    std::uint64_t va,
    std::size_t size_bytes) const {
  auto it = allocations_.upper_bound(va);
  if (it == allocations_.begin()) {
    return nullptr;
  }
  --it;

  const std::uint64_t base_va = it->second.handle.mapped_va;
  if (va < base_va) {
    return nullptr;
  }

  const std::uint64_t offset = va - base_va;
  if (offset > it->second.bytes.size()) {
    return nullptr;
  }
  if (size_bytes > it->second.bytes.size() - static_cast<std::size_t>(offset)) {
    return nullptr;
  }
  return &it->second;
}

std::optional<SingleGpuSimulator::AllocationView>
SingleGpuSimulator::FindAllocationView(std::uint64_t va,
                                       std::size_t size_bytes) {
  AllocationRecord* record = FindAllocation(va, size_bytes);
  if (record == nullptr) {
    return std::nullopt;
  }

  AllocationView view;
  view.record = record;
  view.data = record->bytes.data() +
              static_cast<std::size_t>(va - record->handle.mapped_va);
  return view;
}

std::optional<SingleGpuSimulator::ConstAllocationView>
SingleGpuSimulator::FindAllocationView(std::uint64_t va,
                                       std::size_t size_bytes) const {
  const AllocationRecord* record = FindAllocation(va, size_bytes);
  if (record == nullptr) {
    return std::nullopt;
  }

  ConstAllocationView view;
  view.record = record;
  view.data = record->bytes.data() +
              static_cast<std::size_t>(va - record->handle.mapped_va);
  return view;
}

bool SingleGpuSimulator::LoadDecodedGfx950Program(
    std::uint64_t code_va,
    std::uint32_t code_word_count,
    std::shared_ptr<const std::vector<isa::CompiledInstruction>>* program) {
  if (program == nullptr || code_word_count == 0) {
    return false;
  }

  const std::size_t code_size_bytes =
      static_cast<std::size_t>(code_word_count) * sizeof(std::uint32_t);
  const AllocationRecord* code_record = FindAllocation(code_va, code_size_bytes);
  if (code_record == nullptr) {
    return false;
  }

  const DecodedProgramCacheKey cache_key = {
      .code_va = code_va,
      .code_word_count = code_word_count,
  };
  auto cache_it = decoded_program_cache_.find(cache_key);
  if (cache_it != decoded_program_cache_.end() &&
      cache_it->second.allocation_id == code_record->handle.allocation_id &&
      cache_it->second.allocation_write_version == code_record->write_version &&
      cache_it->second.program != nullptr) {
    ++decode_cache_hits_;
    *program = cache_it->second.program;
    return true;
  }

  const std::size_t code_offset =
      static_cast<std::size_t>(code_va - code_record->handle.mapped_va);
  std::vector<std::uint32_t> code_words(code_word_count, 0);
  std::memcpy(code_words.data(), code_record->bytes.data() + code_offset,
              code_size_bytes);

  isa::Gfx950BinaryDecoder decoder;
  std::vector<isa::DecodedInstruction> decoded_program;
  std::string error_message;
  if (!decoder.DecodeProgram(code_words, &decoded_program, &error_message)) {
    return false;
  }
  auto compiled_program =
      std::make_shared<std::vector<isa::CompiledInstruction>>();
  isa::Gfx950Interpreter interpreter;
  if (!interpreter.CompileProgram(decoded_program, compiled_program.get(),
                                  &error_message)) {
    return false;
  }

  DecodedProgramCacheEntry entry;
  entry.allocation_id = code_record->handle.allocation_id;
  entry.allocation_write_version = code_record->write_version;
  entry.program = compiled_program;
  decoded_program_cache_[cache_key] = entry;
  ++decode_cache_misses_;
  *program = std::move(compiled_program);
  return true;
}

exec::CompletionRecord SingleGpuSimulator::ExecuteDispatch(
    std::uint64_t dispatch_id,
    const exec::SyntheticDispatchPacket& packet) {
  exec::CompletionRecord completion;
  completion.dispatch_id = dispatch_id;
  completion.completed = true;

  switch (packet.opcode) {
    case exec::SyntheticKernelOpcode::kNop:
      completion.success = true;
      break;
    case exec::SyntheticKernelOpcode::kFill32:
      completion.success = ExecuteFill32(packet);
      break;
    case exec::SyntheticKernelOpcode::kVectorAddI32:
      completion.success = ExecuteVectorAddI32(packet);
      break;
    case exec::SyntheticKernelOpcode::kGfx950Program:
      completion.success = ExecuteGfx950Program(packet);
      break;
  }

  return completion;
}

bool SingleGpuSimulator::ExecuteFill32(
    const exec::SyntheticDispatchPacket& packet) {
  const std::size_t total_bytes =
      static_cast<std::size_t>(packet.args.element_count) * sizeof(std::uint32_t);
  AllocationRecord* record = FindAllocation(packet.args.dst_va, total_bytes);
  if (record == nullptr) {
    return false;
  }

  const std::size_t offset =
      static_cast<std::size_t>(packet.args.dst_va - record->handle.mapped_va);
  for (std::uint32_t index = 0; index < packet.args.element_count; ++index) {
    std::memcpy(record->bytes.data() + offset + (index * sizeof(std::uint32_t)),
                &packet.args.immediate_u32, sizeof(packet.args.immediate_u32));
  }
  return true;
}

bool SingleGpuSimulator::ExecuteVectorAddI32(
    const exec::SyntheticDispatchPacket& packet) {
  const std::size_t total_bytes =
      static_cast<std::size_t>(packet.args.element_count) * sizeof(std::int32_t);
  const AllocationRecord* lhs = FindAllocation(packet.args.src0_va, total_bytes);
  const AllocationRecord* rhs = FindAllocation(packet.args.src1_va, total_bytes);
  AllocationRecord* dst = FindAllocation(packet.args.dst_va, total_bytes);
  if (lhs == nullptr || rhs == nullptr || dst == nullptr) {
    return false;
  }

  const std::size_t lhs_offset =
      static_cast<std::size_t>(packet.args.src0_va - lhs->handle.mapped_va);
  const std::size_t rhs_offset =
      static_cast<std::size_t>(packet.args.src1_va - rhs->handle.mapped_va);
  const std::size_t dst_offset =
      static_cast<std::size_t>(packet.args.dst_va - dst->handle.mapped_va);

  for (std::uint32_t index = 0; index < packet.args.element_count; ++index) {
    std::int32_t lhs_value = 0;
    std::int32_t rhs_value = 0;
    std::int32_t result = 0;
    const std::size_t byte_offset = index * sizeof(std::int32_t);

    std::memcpy(&lhs_value, lhs->bytes.data() + lhs_offset + byte_offset,
                sizeof(lhs_value));
    std::memcpy(&rhs_value, rhs->bytes.data() + rhs_offset + byte_offset,
                sizeof(rhs_value));
    result = lhs_value + rhs_value;
    std::memcpy(dst->bytes.data() + dst_offset + byte_offset, &result,
                sizeof(result));
  }
  return true;
}

bool SingleGpuSimulator::ExecuteGfx950Program(
    const exec::SyntheticDispatchPacket& packet) {
  if (packet.args.code_word_count == 0 || packet.args.wave_count == 0) {
    return false;
  }

  std::string error_message;
  std::shared_ptr<const std::vector<isa::CompiledInstruction>> program;
  if (!LoadDecodedGfx950Program(packet.args.code_va, packet.args.code_word_count,
                                &program) ||
      program == nullptr) {
    return false;
  }

  if (packet.args.sgpr_state_count > isa::WaveExecutionState::kScalarRegisterCount ||
      packet.args.vgpr_state_count > isa::WaveExecutionState::kVectorRegisterCount) {
    return false;
  }

  isa::Gfx950Interpreter interpreter;
  SimulatorExecutionMemory memory(this);
  std::vector<isa::WaveExecutionState> waves(packet.args.wave_count);
  std::vector<std::uint64_t> exec_masks(packet.args.wave_count, packet.args.exec_mask);
  if (packet.args.exec_mask_va != 0) {
    const auto exec_mask_view = FindAllocationView(
        packet.args.exec_mask_va, exec_masks.size() * sizeof(exec_masks[0]));
    if (!exec_mask_view.has_value()) {
      return false;
    }
    std::memcpy(exec_masks.data(), exec_mask_view->data,
                exec_masks.size() * sizeof(exec_masks[0]));
  }

  const std::size_t sgpr_wave_bytes =
      static_cast<std::size_t>(packet.args.sgpr_state_count) *
      sizeof(waves.front().sgprs[0]);
  const std::size_t vgpr_wave_bytes =
      static_cast<std::size_t>(packet.args.vgpr_state_count) *
      isa::WaveExecutionState::kLaneCount * sizeof(waves.front().vgprs[0][0]);
  std::optional<AllocationView> sgpr_state_view;
  if (packet.args.sgpr_state_count != 0) {
    sgpr_state_view = FindAllocationView(packet.args.sgpr_state_va,
                                         waves.size() * sgpr_wave_bytes);
    if (!sgpr_state_view.has_value()) {
      return false;
    }
  }
  std::optional<AllocationView> vgpr_state_view;
  if (packet.args.vgpr_state_count != 0) {
    vgpr_state_view = FindAllocationView(packet.args.vgpr_state_va,
                                         waves.size() * vgpr_wave_bytes);
    if (!vgpr_state_view.has_value()) {
      return false;
    }
  }
  for (std::size_t wave_index = 0; wave_index < waves.size(); ++wave_index) {
    isa::WaveExecutionState& state = waves[wave_index];
    state.exec_mask = exec_masks[wave_index];
    state.workgroup_wave_count = packet.args.wave_count;
    if (packet.args.sgpr_state_count != 0) {
      std::memcpy(state.sgprs.data(),
                  sgpr_state_view->data + wave_index * sgpr_wave_bytes,
                  sgpr_wave_bytes);
    }
    if (packet.args.vgpr_state_count != 0) {
      std::memcpy(state.vgprs.data(),
                  vgpr_state_view->data + wave_index * vgpr_wave_bytes,
                  vgpr_wave_bytes);
    }
  }

  isa::WorkgroupExecutionContext workgroup;
  if (waves.size() > 1) {
    shared_lds_scratch_.resize(isa::WaveExecutionState::kLdsSizeBytes);
    std::memset(shared_lds_scratch_.data(), 0, shared_lds_scratch_.size());
    workgroup.shared_lds = std::span<std::byte>(shared_lds_scratch_.data(),
                                                shared_lds_scratch_.size());
    workgroup.wave_count = packet.args.wave_count;
  }

  while (true) {
    bool all_done = true;
    std::size_t blocked_waves = 0;
    for (isa::WaveExecutionState& state : waves) {
      if (state.halted || state.pc >= program->size()) {
        continue;
      }
      all_done = false;
      if (state.waiting_on_barrier) {
        ++blocked_waves;
        continue;
      }

      isa::ProgramRunState run_state = isa::ProgramRunState::kCompleted;
      const isa::WorkgroupExecutionContext* workgroup_ptr =
          waves.size() > 1 ? &workgroup : nullptr;
      if (!interpreter.ExecuteProgramUntilYield(
              *program, &state, &memory, workgroup_ptr, &run_state,
              &error_message)) {
        return false;
      }
      if (run_state == isa::ProgramRunState::kBlockedOnBarrier) {
        ++blocked_waves;
      }
    }

    if (all_done) {
      break;
    }
    if (blocked_waves != 0) {
      if (blocked_waves != waves.size()) {
        return false;
      }
      for (isa::WaveExecutionState& state : waves) {
        state.waiting_on_barrier = false;
      }
    }
  }

  for (std::size_t wave_index = 0; wave_index < waves.size(); ++wave_index) {
    const isa::WaveExecutionState& state = waves[wave_index];
    if (packet.args.sgpr_state_count != 0) {
      std::memcpy(sgpr_state_view->data + wave_index * sgpr_wave_bytes,
                  state.sgprs.data(), sgpr_wave_bytes);
    }
    if (packet.args.vgpr_state_count != 0) {
      std::memcpy(vgpr_state_view->data + wave_index * vgpr_wave_bytes,
                  state.vgprs.data(), vgpr_wave_bytes);
    }
  }
  if (sgpr_state_view.has_value()) {
    ++sgpr_state_view->record->write_version;
  }
  if (vgpr_state_view.has_value()) {
    ++vgpr_state_view->record->write_version;
  }
  return true;
}

}  // namespace mirage::sim
