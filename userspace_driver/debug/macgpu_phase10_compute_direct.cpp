// Standalone libmacgpu version of try_phase10_compute_direct.py.
//
// This deliberately avoids ROCr. Use it after a clean phase9 bring-up to
// isolate libmacgpu/DEEXT MMIO behavior from ROCr direct queue activation.

#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "macgpu.h"

namespace {

constexpr uint32_t kMmioBar = 5;
constexpr uint32_t kVramBar = 0;
constexpr uint32_t kDoorbellBar = 2;

constexpr uint32_t GC_B0 = 0x1260;
constexpr uint32_t GC_B1 = 0xA000;

constexpr uint32_t regGRBM_GFX_CNTL = 0x0900;
constexpr uint32_t regCP_MQD_BASE_ADDR = 0x1fa9;
constexpr uint32_t regCP_MQD_BASE_ADDR_HI = 0x1faa;
constexpr uint32_t regCP_HQD_ACTIVE = 0x1fab;
constexpr uint32_t regCP_HQD_VMID = 0x1fac;
constexpr uint32_t regCP_HQD_PERSISTENT_STATE = 0x1fad;
constexpr uint32_t regCP_HQD_PQ_BASE = 0x1fb1;
constexpr uint32_t regCP_HQD_PQ_BASE_HI = 0x1fb2;
constexpr uint32_t regCP_HQD_PQ_RPTR = 0x1fb3;
constexpr uint32_t regCP_HQD_PQ_RPTR_REPORT_ADDR = 0x1fb4;
constexpr uint32_t regCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x1fb5;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR = 0x1fb6;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x1fb7;
constexpr uint32_t regCP_HQD_PQ_DOORBELL_CONTROL = 0x1fb8;
constexpr uint32_t regCP_HQD_PQ_CONTROL = 0x1fba;
constexpr uint32_t regCP_HQD_DEQUEUE_REQUEST = 0x1fc1;
constexpr uint32_t regCP_MQD_CONTROL = 0x1fcb;
constexpr uint32_t regCP_HQD_EOP_BASE_ADDR = 0x1fce;
constexpr uint32_t regCP_HQD_EOP_BASE_ADDR_HI = 0x1fcf;
constexpr uint32_t regCP_HQD_EOP_CONTROL = 0x1fd0;
constexpr uint32_t regCP_HQD_PQ_WPTR_LO = 0x1fdf;
constexpr uint32_t regCP_HQD_PQ_WPTR_HI = 0x1fe0;

constexpr uint32_t regCP_STAT = 0x0f40;
constexpr uint32_t regGRBM_STATUS = 0x0da4;

constexpr uint32_t CP_HQD_PERSISTENT_STATE_DEFAULT = 0x0be05501;

constexpr uint64_t MQD_OFF = 0x1900000;
constexpr uint64_t RING_OFF = 0x1902000;
constexpr uint64_t EOP_OFF = 0x1910000;
constexpr uint64_t RPTR_OFF = 0x1920000;
constexpr uint64_t WPTR_OFF = 0x1921000;

constexpr uint32_t MQD_SIZE = 0x1000;
constexpr uint32_t RING_SIZE = 0x1000;
constexpr uint32_t EOP_SIZE = 0x1000;
constexpr uint32_t COMPUTE_DOORBELL = 0x20;

uint32_t Rd(macgpu_device_t* dev, uint32_t base, uint32_t reg) {
  uint32_t value = 0;
  macgpu_status_t status =
      macgpu_mmio_read32(dev, kMmioBar, static_cast<uint64_t>(base + reg) * 4, &value);
  if (status != MACGPU_SUCCESS) {
    std::fprintf(stderr, "mmio read failed base=0x%x reg=0x%x status=%d\n", base, reg, status);
    std::exit(2);
  }
  return value;
}

void Wr(macgpu_device_t* dev, uint32_t base, uint32_t reg, uint32_t value) {
  macgpu_status_t status =
      macgpu_mmio_write32(dev, kMmioBar, static_cast<uint64_t>(base + reg) * 4, value);
  if (status != MACGPU_SUCCESS) {
    std::fprintf(stderr, "mmio write failed base=0x%x reg=0x%x status=%d\n", base, reg, status);
    std::exit(2);
  }
}

void SelectHqd(macgpu_device_t* dev, uint32_t me, uint32_t pipe, uint32_t queue, uint32_t vmid = 0) {
  Wr(dev, GC_B1, regGRBM_GFX_CNTL,
     ((pipe & 0x3u) << 0) | ((me & 0x3u) << 2) |
         ((vmid & 0xfu) << 4) | ((queue & 0x7u) << 8));
}

void DequeueSelectedHqd(macgpu_device_t* dev, const char* phase) {
  const uint32_t pre_active = Rd(dev, GC_B0, regCP_HQD_ACTIVE);
  Wr(dev, GC_B0, regCP_HQD_DEQUEUE_REQUEST, 1);
  uint32_t active = pre_active;
  for (uint32_t i = 0; i < 1000; ++i) {
    active = Rd(dev, GC_B0, regCP_HQD_ACTIVE);
    if (active == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  Wr(dev, GC_B0, regCP_HQD_DEQUEUE_REQUEST, 0);

  uint32_t doorbell_control = Rd(dev, GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL);
  if (active == 0 && (doorbell_control & 0x40000000u) != 0) {
    Wr(dev, GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, doorbell_control & ~0x40000000u);
    doorbell_control = Rd(dev, GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL);
  }
  std::printf("  %s dequeue: pre_active=0x%x post_active=0x%x doorbell_control=0x%x\n",
              phase, pre_active, active, doorbell_control);
}

uint32_t Low32(uint64_t value) { return static_cast<uint32_t>(value); }
uint32_t High32(uint64_t value) { return static_cast<uint32_t>(value >> 32); }

void VramWr32(void* bar0, uint64_t off, uint32_t value) {
  *reinterpret_cast<volatile uint32_t*>(static_cast<char*>(bar0) + off) = value;
}

void VramWr64(void* bar0, uint64_t off, uint64_t value) {
  *reinterpret_cast<volatile uint64_t*>(static_cast<char*>(bar0) + off) = value;
}

uint32_t VramRd32(void* bar0, uint64_t off) {
  return *reinterpret_cast<volatile uint32_t*>(static_cast<char*>(bar0) + off);
}

uint64_t VramRd64(void* bar0, uint64_t off) {
  return *reinterpret_cast<volatile uint64_t*>(static_cast<char*>(bar0) + off);
}

void DoorbellWr64(void* bar2, uint32_t index, uint64_t value) {
  *reinterpret_cast<volatile uint64_t*>(static_cast<char*>(bar2) + index * 4) = value;
}

void ZeroVram(void* bar0, uint64_t off, uint64_t size) {
  for (uint64_t i = 0; i < size; i += 4) VramWr32(bar0, off + i, 0);
}

}  // namespace

int main() {
  macgpu_device_t* dev = nullptr;
  macgpu_status_t status = macgpu_open(&dev);
  if (status != MACGPU_SUCCESS) {
    std::fprintf(stderr, "macgpu_open failed: %d\n", status);
    return 1;
  }

  macgpu_device_info_t info{};
  status = macgpu_get_info(dev, &info);
  if (status != MACGPU_SUCCESS) {
    std::fprintf(stderr, "macgpu_get_info failed: %d\n", status);
    return 1;
  }
  std::printf("device=0x%04x rev=0x%02x\n", info.device_id, info.revision_id);

  void* bar0 = nullptr;
  void* bar2 = nullptr;
  uint64_t bar0_size = 0;
  uint64_t bar2_size = 0;
  if (macgpu_map_bar(dev, kVramBar, &bar0, &bar0_size) != MACGPU_SUCCESS ||
      macgpu_map_bar(dev, kDoorbellBar, &bar2, &bar2_size) != MACGPU_SUCCESS) {
    std::fprintf(stderr, "BAR mapping failed\n");
    return 1;
  }

  uint32_t fb_reg = 0;
  status = macgpu_mmio_read32(dev, kMmioBar, (0x1A000 + 0x0554) * 4, &fb_reg);
  if (status != MACGPU_SUCCESS) {
    std::fprintf(stderr, "framebuffer base read failed: %d\n", status);
    return 1;
  }
  const uint64_t fb_base = static_cast<uint64_t>(fb_reg & 0xFFFFFFu) << 24;
  std::printf("fb_base=0x%llx BAR2 size=%lluKB\n",
              static_cast<unsigned long long>(fb_base),
              static_cast<unsigned long long>(bar2_size / 1024));

  for (auto [off, size] : {std::pair<uint64_t, uint64_t>{MQD_OFF, MQD_SIZE},
                           {RING_OFF, RING_SIZE},
                           {EOP_OFF, EOP_SIZE},
                           {RPTR_OFF, 0x20},
                           {WPTR_OFF, 0x20}}) {
    ZeroVram(bar0, off, size);
  }

  SelectHqd(dev, 1, 0, 0);
  const uint32_t pre_active = Rd(dev, GC_B0, regCP_HQD_ACTIVE);
  std::printf("target compute HQD me=1 pipe=0 queue=0 PRE_ACTIVE=0x%x\n", pre_active);
  if (pre_active != 0) {
    DequeueSelectedHqd(dev, "pre-existing");
  }

  const uint64_t mqd_mc = fb_base + MQD_OFF;
  const uint64_t ring_mc = fb_base + RING_OFF;
  const uint64_t eop_mc = fb_base + EOP_OFF;
  const uint64_t rptr_mc = fb_base + RPTR_OFF;
  const uint64_t wptr_mc = fb_base + WPTR_OFF;

  uint32_t mqd[MQD_SIZE / 4]{};
  mqd[0] = 0xC0310800;
  mqd[1] = 1;
  for (uint32_t dw : {0x17u, 0x18u, 0x1Au, 0x1Bu}) mqd[dw] = 0xFFFFFFFFu;
  mqd[0x2C] = 7;
  mqd[0xA5] = Low32(eop_mc >> 8);
  mqd[0xA6] = High32(eop_mc >> 8);
  mqd[0xA7] = 9;
  mqd[0x80] = Low32(mqd_mc) & 0xFFFFFFFCu;
  mqd[0x81] = High32(mqd_mc);
  mqd[0x82] = 1;
  mqd[0x84] = (CP_HQD_PERSISTENT_STATE_DEFAULT & ~(0x3FFu << 8)) | (0x55u << 8);
  mqd[0x88] = Low32(ring_mc >> 8);
  mqd[0x89] = High32(ring_mc >> 8);
  mqd[0x8B] = Low32(rptr_mc) & 0xFFFFFFFCu;
  mqd[0x8C] = High32(rptr_mc) & 0xFFFFu;
  mqd[0x8D] = Low32(wptr_mc) & 0xFFFFFFF8u;
  mqd[0x8E] = High32(wptr_mc) & 0xFFFFu;
  mqd[0x8F] = ((COMPUTE_DOORBELL & 0x03FFFFFFu) << 2) | (1u << 30);
  mqd[0x91] = 9 | (5u << 8) | (1u << 27) | (1u << 28) | (1u << 30) |
              (1u << 31) | 0x300000u | 0x8000u;
  mqd[0x95] = 0x00300000;
  mqd[0xA2] = 0x100;
  mqd[0xB8] = 1u << 15;

  for (size_t i = 0; i < sizeof(mqd) / sizeof(mqd[0]); ++i) {
    VramWr32(bar0, MQD_OFF + i * 4, mqd[i]);
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);

  Wr(dev, GC_B0, regCP_HQD_ACTIVE, 0);
  Wr(dev, GC_B0, regCP_HQD_PQ_RPTR, 0);
  Wr(dev, GC_B0, regCP_HQD_PQ_WPTR_LO, 0);
  Wr(dev, GC_B0, regCP_HQD_PQ_WPTR_HI, 0);
  Wr(dev, GC_B0, regCP_HQD_VMID, Rd(dev, GC_B0, regCP_HQD_VMID) & ~0xFu);
  Wr(dev, GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL,
     Rd(dev, GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL) & ~0x40000000u);
  Wr(dev, GC_B0, regCP_MQD_BASE_ADDR, mqd[0x80]);
  Wr(dev, GC_B0, regCP_MQD_BASE_ADDR_HI, mqd[0x81]);
  Wr(dev, GC_B0, regCP_MQD_CONTROL, 0);
  Wr(dev, GC_B0, regCP_HQD_EOP_BASE_ADDR, mqd[0xA5]);
  Wr(dev, GC_B0, regCP_HQD_EOP_BASE_ADDR_HI, mqd[0xA6]);
  Wr(dev, GC_B0, regCP_HQD_EOP_CONTROL, mqd[0xA7]);
  Wr(dev, GC_B0, regCP_HQD_PQ_BASE, mqd[0x88]);
  Wr(dev, GC_B0, regCP_HQD_PQ_BASE_HI, mqd[0x89]);
  Wr(dev, GC_B0, regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B]);
  Wr(dev, GC_B0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C]);
  Wr(dev, GC_B0, regCP_HQD_PQ_CONTROL, mqd[0x91]);
  Wr(dev, GC_B0, regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D]);
  Wr(dev, GC_B0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E]);
  Wr(dev, GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F]);
  Wr(dev, GC_B0, regCP_HQD_PERSISTENT_STATE, mqd[0x84]);
  Wr(dev, GC_B0, regCP_HQD_ACTIVE, 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  std::printf("HQD readback after program:\n");
  std::printf("  ACTIVE=0x%08x PQ_BASE=0x%08x:%08x PQ_CONTROL=0x%08x DOORBELL_CONTROL=0x%08x\n",
              Rd(dev, GC_B0, regCP_HQD_ACTIVE),
              Rd(dev, GC_B0, regCP_HQD_PQ_BASE_HI),
              Rd(dev, GC_B0, regCP_HQD_PQ_BASE),
              Rd(dev, GC_B0, regCP_HQD_PQ_CONTROL),
              Rd(dev, GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL));
  std::printf("  RPTR=0x%08x WPTR=0x%08x:%08x\n",
              Rd(dev, GC_B0, regCP_HQD_PQ_RPTR),
              Rd(dev, GC_B0, regCP_HQD_PQ_WPTR_LO),
              Rd(dev, GC_B0, regCP_HQD_PQ_WPTR_HI));

  VramWr32(bar0, RING_OFF, 0xC0001000);
  VramWr32(bar0, RING_OFF + 4, 0);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  VramWr64(bar0, WPTR_OFF, 2);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  std::printf("  BAR readback ring[0]=0x%08x ring[1]=0x%08x wptr_mem=0x%llx\n",
              VramRd32(bar0, RING_OFF), VramRd32(bar0, RING_OFF + 4),
              static_cast<unsigned long long>(VramRd64(bar0, WPTR_OFF)));
  DoorbellWr64(bar2, COMPUTE_DOORBELL, 2);
  std::atomic_thread_fence(std::memory_order_seq_cst);

  std::printf("\n== compute PM4 NOP doorbell index=0x%x wptr=2 ==\n", COMPUTE_DOORBELL);
  bool consumed = false;
  for (uint32_t i = 0; i < 100; ++i) {
    const uint32_t rptr = Rd(dev, GC_B0, regCP_HQD_PQ_RPTR);
    const uint32_t wptr = Rd(dev, GC_B0, regCP_HQD_PQ_WPTR_LO);
    const uint32_t active = Rd(dev, GC_B0, regCP_HQD_ACTIVE);
    if (i == 0 || rptr == 2) {
      std::printf("  rptr=0x%x wptr=0x%x active=0x%x CP_STAT=0x%x GRBM=0x%x\n",
                  rptr, wptr, active, Rd(dev, GC_B0, regCP_STAT),
                  Rd(dev, GC_B0, regGRBM_STATUS));
    }
    if (rptr == 2) {
      consumed = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  if (consumed) {
    std::printf("  COMPUTE QUEUE CONSUMED PM4 NOP \342\234\223\n");
    DequeueSelectedHqd(dev, "cleanup");
  } else {
    std::printf("  compute queue did not consume PM4 before timeout\n");
  }
  Wr(dev, GC_B1, regGRBM_GFX_CNTL, 0);
  macgpu_close(dev);
  return consumed ? 0 : 1;
}
