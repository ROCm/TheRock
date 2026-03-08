/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * amdgpu_lite.h - Shared header for amdgpu_lite kernel module
 *
 * Minimal GPU access primitives for userspace ROCm driver.
 * This header is shared between kernel module and userspace.
 */

#ifndef _AMDGPU_LITE_H_
#define _AMDGPU_LITE_H_

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/ioctl.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
typedef uint8_t  __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef uint64_t __u64;
typedef int32_t  __s32;
typedef int64_t  __s64;
#endif

#define AMDGPU_LITE_MAX_BARS    6

/* ======================================================================
 * BAR info (returned in GET_INFO)
 * ====================================================================== */

struct amdgpu_lite_bar_info {
	__u64 phys_addr;
	__u64 size;
	__u32 is_memory;
	__u32 is_64bit;
	__u32 is_prefetchable;
	__u32 bar_index;        /* PCI BAR index */
};

/* ======================================================================
 * GET_INFO - Return device identification and BAR layout
 * ====================================================================== */

struct amdgpu_lite_get_info {
	/* Output */
	__u16 vendor_id;
	__u16 device_id;
	__u16 subsystem_vendor_id;
	__u16 subsystem_id;
	__u8  revision_id;
	__u8  reserved1[3];
	__u32 num_bars;
	struct amdgpu_lite_bar_info bars[AMDGPU_LITE_MAX_BARS];
	__u64 vram_size;
	__u64 visible_vram_size;
	__u32 mmio_bar_index;       /* Index into bars[] for MMIO registers */
	__u32 vram_bar_index;       /* Index into bars[] for VRAM aperture */
	__u32 doorbell_bar_index;   /* Index into bars[] for doorbell */
	__u32 reserved2;
	__u64 gart_table_bus_addr;  /* Physical address of GART page table */
	__u64 gart_table_size;      /* Size in bytes */
	__u64 gart_gpu_va_start;    /* GPU VA where GART aperture begins */
};

/* ======================================================================
 * MAP_BAR - Set up mmap offset for a specific BAR
 * ====================================================================== */

struct amdgpu_lite_map_bar {
	/* Input */
	__u32 bar_index;        /* Which BAR to map (index into bars[]) */
	__u32 reserved1;
	__u64 offset;           /* Offset within BAR */
	__u64 size;             /* Size to map (0 = entire BAR) */
	/* Output */
	__u64 mmap_offset;      /* Offset to pass to mmap() */
	__u32 reserved2[4];
};

/* ======================================================================
 * ALLOC_GTT - Allocate DMA-coherent system memory
 * ====================================================================== */

struct amdgpu_lite_alloc_gtt {
	/* Input */
	__u64 size;             /* Requested size in bytes */
	__u32 reserved1[2];
	/* Output */
	__u64 handle;           /* Allocation handle for FREE/mmap */
	__u64 bus_addr;         /* DMA bus address (for GPU programming) */
	__u64 mmap_offset;      /* Offset to pass to mmap() */
	__u32 reserved2[4];
};

/* ======================================================================
 * FREE_GTT - Free DMA-coherent system memory
 * ====================================================================== */

struct amdgpu_lite_free_gtt {
	/* Input */
	__u64 handle;           /* Handle from ALLOC_GTT */
	__u32 reserved[4];
};

/* ======================================================================
 * ALLOC_VRAM / FREE_VRAM - VRAM allocation (stub)
 * ====================================================================== */

struct amdgpu_lite_alloc_vram {
	/* Input */
	__u64 size;
	__u32 flags;
	__u32 reserved1;
	/* Output */
	__u64 handle;
	__u64 gpu_addr;
	__u64 mmap_offset;
	__u32 reserved2[4];
};

struct amdgpu_lite_free_vram {
	/* Input */
	__u64 handle;
	__u32 reserved[4];
};

/* ======================================================================
 * MAP_GPU / UNMAP_GPU - GPU page table programming (stub)
 * ====================================================================== */

struct amdgpu_lite_map_gpu {
	/* Input */
	__u64 handle;           /* GTT or VRAM handle */
	__u64 gpu_va;           /* Desired GPU virtual address (0 = auto) */
	__u64 size;
	__u32 flags;
	__u32 reserved[3];
	/* Output */
	__u64 mapped_gpu_va;    /* Actual GPU VA assigned */
};

struct amdgpu_lite_unmap_gpu {
	/* Input */
	__u64 gpu_va;
	__u64 size;
	__u32 reserved[4];
};

/* ======================================================================
 * SETUP_IRQ - Register eventfd for interrupt forwarding (stub)
 * ====================================================================== */

struct amdgpu_lite_setup_irq {
	/* Input */
	__s32 eventfd;          /* eventfd fd (-1 to teardown existing) */
	__u32 irq_source;       /* Interrupt source ID */
	__u32 registration_id;  /* 0 = new registration, nonzero = teardown */
	__u32 reserved[3];
	/* Output (on new registration) */
	__u32 out_registration_id;
	__u32 reserved2;
};

/* ======================================================================
 * Ioctl numbers - type 'L' for Lite
 * ====================================================================== */

#define AMDGPU_LITE_IOC_MAGIC   'L'

#define AMDGPU_LITE_IOC_GET_INFO    _IOR(AMDGPU_LITE_IOC_MAGIC, 0x01, \
					 struct amdgpu_lite_get_info)
#define AMDGPU_LITE_IOC_MAP_BAR     _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x02, \
					   struct amdgpu_lite_map_bar)
#define AMDGPU_LITE_IOC_ALLOC_GTT   _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x10, \
					   struct amdgpu_lite_alloc_gtt)
#define AMDGPU_LITE_IOC_FREE_GTT    _IOW(AMDGPU_LITE_IOC_MAGIC, 0x11, \
					  struct amdgpu_lite_free_gtt)
#define AMDGPU_LITE_IOC_ALLOC_VRAM  _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x20, \
					   struct amdgpu_lite_alloc_vram)
#define AMDGPU_LITE_IOC_FREE_VRAM   _IOW(AMDGPU_LITE_IOC_MAGIC, 0x21, \
					  struct amdgpu_lite_free_vram)
#define AMDGPU_LITE_IOC_MAP_GPU     _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x30, \
					   struct amdgpu_lite_map_gpu)
#define AMDGPU_LITE_IOC_UNMAP_GPU   _IOW(AMDGPU_LITE_IOC_MAGIC, 0x31, \
					  struct amdgpu_lite_unmap_gpu)
#define AMDGPU_LITE_IOC_SETUP_IRQ   _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x40, \
					   struct amdgpu_lite_setup_irq)

/* ======================================================================
 * Mmap offset encoding
 *
 * We use the upper bits of the mmap offset to distinguish BAR mappings
 * from GTT allocations:
 *   [63:60] = type (0=BAR, 1=GTT, 2=VRAM)
 *   [59:40] = device/bar index or allocation handle
 *   [39:0]  = offset within region
 * ====================================================================== */

#define AMDGPU_LITE_MMAP_TYPE_SHIFT   60
#define AMDGPU_LITE_MMAP_TYPE_BAR     0ULL
#define AMDGPU_LITE_MMAP_TYPE_GTT     1ULL
#define AMDGPU_LITE_MMAP_TYPE_VRAM    2ULL

/* ======================================================================
 * Kernel-only declarations
 * ====================================================================== */

#ifdef __KERNEL__

#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/miscdevice.h>
#include <linux/idr.h>
#include <linux/mutex.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/eventfd.h>

#define AMDGPU_LITE_MAX_GTT_ALLOCS  256
#define AMDGPU_LITE_MAX_EVENTS      64
#define AMDGPU_LITE_NAME            "amdgpu_lite"

/* GART table configuration */
#define AMDGPU_LITE_GART_TABLE_SIZE   (1024 * 1024)       /* 1MB */
#define AMDGPU_LITE_GART_NUM_ENTRIES  (AMDGPU_LITE_GART_TABLE_SIZE / 8)  /* 128K entries */
#define AMDGPU_LITE_GART_VA_START     0x100000000ULL       /* 4GB, after VRAM */

/* VRAM allocation tracking */
struct vram_allocation {
	struct list_head list;
	uint64_t handle;
	uint64_t gpu_offset;    /* Offset within VRAM */
	uint64_t size;
	uint32_t num_pages;
};

/* VRAM bitmap allocator */
struct vram_allocator {
	unsigned long *bitmap;  /* 1 bit per VRAM page */
	uint64_t total_pages;
	uint64_t free_pages;
	struct mutex lock;
};

/* Per-BAR kernel state */
struct amdgpu_lite_bar {
	resource_size_t phys_addr;
	resource_size_t size;
	void __iomem *kaddr;        /* iomap kernel address (NULL if not mapped) */
	unsigned int bar_index;     /* PCI BAR index */
	bool is_memory;
	bool is_64bit;
	bool is_prefetchable;
};

/* Eventfd registration for interrupt forwarding */
struct amdgpu_lite_event_reg {
	struct eventfd_ctx *ctx;
	uint32_t irq_source;
	struct amdgpu_lite_fpriv *owner;  /* For cleanup on file close */
	bool in_use;
};

/* GTT (DMA-coherent) allocation tracking */
struct amdgpu_lite_gtt_alloc {
	struct list_head list;
	void *cpu_addr;
	dma_addr_t bus_addr;
	size_t size;
	u64 handle;
	u64 mmap_offset;
};

/* Per-open-file state (for cleanup on close) */
struct amdgpu_lite_fpriv {
	struct amdgpu_lite_device *ldev;
	struct list_head gtt_allocs;
	struct list_head vram_allocs;
	struct mutex alloc_lock;
	u64 next_gtt_handle;
	u64 next_vram_handle;
};

/* Per-device state */
struct amdgpu_lite_device {
	struct pci_dev *pdev;
	struct miscdevice misc;
	char misc_name[32];

	/* PCI BARs */
	struct amdgpu_lite_bar bars[AMDGPU_LITE_MAX_BARS];
	unsigned int num_bars;

	/* BAR classification (same logic as WDDM driver) */
	unsigned int mmio_bar_idx;      /* Smallest BAR (~512KB MMIO regs) */
	unsigned int vram_bar_idx;      /* Largest BAR (VRAM aperture) */
	unsigned int doorbell_bar_idx;  /* Middle BAR (doorbell) */

	/* VRAM info */
	u64 vram_size;
	u64 visible_vram_size;

	/* VRAM allocator */
	struct vram_allocator vram;

	/* GART page table */
	uint64_t *gart_table;          /* GART PTE array (DMA coherent) */
	dma_addr_t gart_table_bus_addr;
	uint64_t gart_size;            /* Size of GART table in bytes */
	uint64_t gart_next_gpu_va;     /* Bump allocator for auto-assign */
	struct mutex gart_lock;

	/* Interrupt / eventfd */
	struct amdgpu_lite_event_reg events[AMDGPU_LITE_MAX_EVENTS];
	spinlock_t events_lock;
	int irq_vector;                /* MSI-X/MSI vector number */

	/* Device tracking */
	int index;
	struct list_head list;
};

/* Implemented in pci_setup.c */
int amdgpu_lite_pci_setup(struct amdgpu_lite_device *ldev);
void amdgpu_lite_pci_cleanup(struct amdgpu_lite_device *ldev);
long amdgpu_lite_ioctl_get_info(struct amdgpu_lite_device *ldev,
				unsigned long arg);
long amdgpu_lite_ioctl_map_bar(struct amdgpu_lite_device *ldev,
			       unsigned long arg);
int amdgpu_lite_mmap_bar(struct amdgpu_lite_device *ldev,
			 struct vm_area_struct *vma);

/* Implemented in memory.c */
long amdgpu_lite_ioctl_alloc_gtt(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg);
long amdgpu_lite_ioctl_free_gtt(struct amdgpu_lite_fpriv *fpriv,
				unsigned long arg);
long amdgpu_lite_ioctl_alloc_vram(struct amdgpu_lite_fpriv *fpriv,
				  unsigned long arg);
long amdgpu_lite_ioctl_free_vram(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg);
long amdgpu_lite_ioctl_map_gpu(struct amdgpu_lite_fpriv *fpriv,
			       unsigned long arg);
long amdgpu_lite_ioctl_unmap_gpu(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg);
int amdgpu_lite_mmap_gtt(struct amdgpu_lite_fpriv *fpriv,
			 struct vm_area_struct *vma);
int amdgpu_lite_mmap_vram(struct amdgpu_lite_fpriv *fpriv,
			  struct vm_area_struct *vma);
void amdgpu_lite_free_all_gtt(struct amdgpu_lite_fpriv *fpriv);
void amdgpu_lite_free_all_vram(struct amdgpu_lite_fpriv *fpriv);
int vram_allocator_init(struct amdgpu_lite_device *ldev);
void vram_allocator_destroy(struct amdgpu_lite_device *ldev);

/* Implemented in memory.c - GART */
int amdgpu_lite_gart_init(struct amdgpu_lite_device *ldev);
void amdgpu_lite_gart_cleanup(struct amdgpu_lite_device *ldev);

/* Implemented in irq.c */
int amdgpu_lite_irq_init(struct amdgpu_lite_device *ldev);
void amdgpu_lite_irq_cleanup(struct amdgpu_lite_device *ldev);
long amdgpu_lite_ioctl_setup_irq(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg);
void amdgpu_lite_irq_release_fpriv(struct amdgpu_lite_fpriv *fpriv);

#endif /* __KERNEL__ */
#endif /* _AMDGPU_LITE_H_ */
