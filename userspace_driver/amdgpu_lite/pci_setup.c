// SPDX-License-Identifier: GPL-2.0
/*
 * pci_setup.c - PCI and BAR management for amdgpu_lite
 *
 * Handles PCI device enable, BAR mapping, BAR classification,
 * and the GET_INFO / MAP_BAR ioctls.
 */

#include <linux/pci.h>
#include <linux/uaccess.h>
#include <linux/io.h>

#include "amdgpu_lite.h"

/*
 * RCC_CONFIG_MEMSIZE register - reports VRAM size in MB.
 * Same offset used in the WDDM driver.
 */
#define RCC_CONFIG_MEMSIZE_REG      0x378C  /* 0xDE3 * 4 */

/* ======================================================================
 * BAR enumeration
 * ====================================================================== */

static void enumerate_bars(struct amdgpu_lite_device *ldev)
{
	struct pci_dev *pdev = ldev->pdev;
	unsigned int i;
	unsigned int count = 0;

	for (i = 0; i < AMDGPU_LITE_MAX_BARS && count < AMDGPU_LITE_MAX_BARS; i++) {
		resource_size_t start = pci_resource_start(pdev, i);
		resource_size_t len = pci_resource_len(pdev, i);
		unsigned long flags = pci_resource_flags(pdev, i);

		if (len == 0)
			continue;

		if (!(flags & IORESOURCE_MEM))
			continue;

		ldev->bars[count].phys_addr = start;
		ldev->bars[count].size = len;
		ldev->bars[count].bar_index = i;
		ldev->bars[count].is_memory = true;
		ldev->bars[count].is_64bit = !!(flags & IORESOURCE_MEM_64);
		ldev->bars[count].is_prefetchable = !!(flags & IORESOURCE_PREFETCH);
		ldev->bars[count].kaddr = NULL;

		dev_info(&pdev->dev,
			 "amdgpu_lite: BAR[%u] (PCI BAR%u) phys=0x%llx size=%llu%s%s\n",
			 count, i,
			 (u64)start, (u64)len,
			 (flags & IORESOURCE_PREFETCH) ? " prefetchable" : "",
			 (flags & IORESOURCE_MEM_64) ? " 64bit" : "");

		count++;

		/* 64-bit BARs consume two BAR slots */
		if (flags & IORESOURCE_MEM_64)
			i++;
	}

	ldev->num_bars = count;
}

/*
 * classify_bars - Determine which BAR is MMIO, VRAM, and doorbell.
 *
 * AMD GPUs present BARs as:
 *   BAR0/1: VRAM aperture (largest, prefetchable, 64-bit)
 *   BAR2/3: Doorbell aperture (medium, prefetchable, 64-bit, ~256MB)
 *   BAR5:   MMIO registers (smallest, non-prefetchable, 32-bit, ~512KB)
 *
 * We classify by size: VRAM=largest, MMIO=smallest, Doorbell=middle.
 */
static void classify_bars(struct amdgpu_lite_device *ldev)
{
	unsigned int i;
	resource_size_t largest_size = 0;
	resource_size_t smallest_size = ~(resource_size_t)0;
	unsigned int largest_idx = 0, smallest_idx = 0;

	ldev->mmio_bar_idx = 0;
	ldev->vram_bar_idx = 0;
	ldev->doorbell_bar_idx = 0;

	if (ldev->num_bars < 2)
		return;

	for (i = 0; i < ldev->num_bars; i++) {
		if (!ldev->bars[i].is_memory || ldev->bars[i].size == 0)
			continue;
		if (ldev->bars[i].size > largest_size) {
			largest_size = ldev->bars[i].size;
			largest_idx = i;
		}
		if (ldev->bars[i].size < smallest_size) {
			smallest_size = ldev->bars[i].size;
			smallest_idx = i;
		}
	}

	ldev->vram_bar_idx = largest_idx;
	ldev->mmio_bar_idx = smallest_idx;

	/* Doorbell is the one that's neither largest nor smallest */
	for (i = 0; i < ldev->num_bars; i++) {
		if (!ldev->bars[i].is_memory || ldev->bars[i].size == 0)
			continue;
		if (i != largest_idx && i != smallest_idx) {
			ldev->doorbell_bar_idx = i;
			break;
		}
	}

	/* If only 2 BARs, MMIO is smaller, doorbell is larger */
	if (ldev->num_bars == 2) {
		ldev->mmio_bar_idx = smallest_idx;
		ldev->doorbell_bar_idx = largest_idx;
	}

	dev_info(&ldev->pdev->dev,
		 "amdgpu_lite: BAR classification: MMIO=%u (%lluKB) VRAM=%u (%lluMB) Doorbell=%u (%lluMB)\n",
		 ldev->mmio_bar_idx,
		 (u64)ldev->bars[ldev->mmio_bar_idx].size / 1024,
		 ldev->vram_bar_idx,
		 (u64)ldev->bars[ldev->vram_bar_idx].size / (1024 * 1024),
		 ldev->doorbell_bar_idx,
		 (u64)ldev->bars[ldev->doorbell_bar_idx].size / (1024 * 1024));
}

/*
 * detect_vram_size - Read VRAM size from MMIO register or fall back to BAR size.
 */
static void detect_vram_size(struct amdgpu_lite_device *ldev)
{
	struct amdgpu_lite_bar *mmio_bar = &ldev->bars[ldev->mmio_bar_idx];
	struct amdgpu_lite_bar *vram_bar = &ldev->bars[ldev->vram_bar_idx];
	u32 mem_size_mb;

	if (!mmio_bar->kaddr ||
	    RCC_CONFIG_MEMSIZE_REG + sizeof(u32) > mmio_bar->size) {
		ldev->vram_size = vram_bar->size;
		ldev->visible_vram_size = vram_bar->size;
		return;
	}

	mem_size_mb = ioread32(mmio_bar->kaddr + RCC_CONFIG_MEMSIZE_REG);

	dev_info(&ldev->pdev->dev,
		 "amdgpu_lite: RCC_CONFIG_MEMSIZE = 0x%08x (%u MB)\n",
		 mem_size_mb, mem_size_mb);

	if (mem_size_mb > 0 && mem_size_mb < 0x100000) {
		ldev->vram_size = (u64)mem_size_mb * 1024ULL * 1024ULL;
	} else {
		ldev->vram_size = vram_bar->size;
	}

	ldev->visible_vram_size = min(ldev->vram_size, (u64)vram_bar->size);
}

/* ======================================================================
 * PCI setup / cleanup
 * ====================================================================== */

int amdgpu_lite_pci_setup(struct amdgpu_lite_device *ldev)
{
	struct pci_dev *pdev = ldev->pdev;
	int ret;

	ret = pci_enable_device(pdev);
	if (ret) {
		dev_err(&pdev->dev, "amdgpu_lite: pci_enable_device failed: %d\n", ret);
		return ret;
	}

	ret = pci_request_regions(pdev, AMDGPU_LITE_NAME);
	if (ret) {
		dev_err(&pdev->dev, "amdgpu_lite: pci_request_regions failed: %d\n", ret);
		goto err_disable;
	}

	pci_set_master(pdev);

	/* Set DMA mask for 64-bit addressing */
	ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
	if (ret) {
		ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
		if (ret) {
			dev_err(&pdev->dev, "amdgpu_lite: no usable DMA configuration\n");
			goto err_regions;
		}
	}

	/* Enumerate and classify BARs */
	enumerate_bars(ldev);
	classify_bars(ldev);

	/* Map MMIO BAR for register access */
	if (ldev->num_bars > 0) {
		struct amdgpu_lite_bar *mmio = &ldev->bars[ldev->mmio_bar_idx];

		mmio->kaddr = pci_iomap(pdev, mmio->bar_index, 0);
		if (!mmio->kaddr) {
			dev_warn(&pdev->dev,
				 "amdgpu_lite: failed to iomap MMIO BAR%u\n",
				 mmio->bar_index);
			/* Non-fatal: userspace can still mmap BARs directly */
		}
	}

	detect_vram_size(ldev);

	return 0;

err_regions:
	pci_release_regions(pdev);
err_disable:
	pci_disable_device(pdev);
	return ret;
}

void amdgpu_lite_pci_cleanup(struct amdgpu_lite_device *ldev)
{
	struct pci_dev *pdev = ldev->pdev;
	unsigned int i;

	/* Unmap any iomap'd BARs */
	for (i = 0; i < ldev->num_bars; i++) {
		if (ldev->bars[i].kaddr) {
			pci_iounmap(pdev, ldev->bars[i].kaddr);
			ldev->bars[i].kaddr = NULL;
		}
	}

	pci_release_regions(pdev);
	pci_disable_device(pdev);
}

/* ======================================================================
 * GET_INFO ioctl
 * ====================================================================== */

long amdgpu_lite_ioctl_get_info(struct amdgpu_lite_device *ldev,
				unsigned long arg)
{
	struct amdgpu_lite_get_info info;
	struct pci_dev *pdev = ldev->pdev;
	unsigned int i;

	memset(&info, 0, sizeof(info));

	info.vendor_id = pdev->vendor;
	info.device_id = pdev->device;
	info.subsystem_vendor_id = pdev->subsystem_vendor;
	info.subsystem_id = pdev->subsystem_device;
	info.revision_id = pdev->revision;
	info.num_bars = ldev->num_bars;

	for (i = 0; i < ldev->num_bars && i < AMDGPU_LITE_MAX_BARS; i++) {
		info.bars[i].phys_addr = ldev->bars[i].phys_addr;
		info.bars[i].size = ldev->bars[i].size;
		info.bars[i].is_memory = ldev->bars[i].is_memory;
		info.bars[i].is_64bit = ldev->bars[i].is_64bit;
		info.bars[i].is_prefetchable = ldev->bars[i].is_prefetchable;
		info.bars[i].bar_index = ldev->bars[i].bar_index;
	}

	info.vram_size = ldev->vram_size;
	info.visible_vram_size = ldev->visible_vram_size;
	info.mmio_bar_index = ldev->mmio_bar_idx;
	info.vram_bar_index = ldev->vram_bar_idx;
	info.doorbell_bar_index = ldev->doorbell_bar_idx;

	/* GART page table info */
	info.gart_table_bus_addr = ldev->gart_table_bus_addr;
	info.gart_table_size = ldev->gart_size;
	info.gart_gpu_va_start = AMDGPU_LITE_GART_VA_START;

	if (copy_to_user((void __user *)arg, &info, sizeof(info)))
		return -EFAULT;

	return 0;
}

/* ======================================================================
 * MAP_BAR ioctl - compute mmap offset for a BAR
 * ====================================================================== */

long amdgpu_lite_ioctl_map_bar(struct amdgpu_lite_device *ldev,
			       unsigned long arg)
{
	struct amdgpu_lite_map_bar params;

	if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
		return -EFAULT;

	if (params.bar_index >= ldev->num_bars)
		return -EINVAL;

	/* Encode the mmap offset: type=BAR, bar index in middle bits */
	params.mmap_offset = (AMDGPU_LITE_MMAP_TYPE_BAR << AMDGPU_LITE_MMAP_TYPE_SHIFT) |
			     ((u64)params.bar_index << 40) |
			     (params.offset & 0xFFFFFFFFFFULL);

	if (copy_to_user((void __user *)arg, &params, sizeof(params)))
		return -EFAULT;

	return 0;
}

/* ======================================================================
 * mmap handler for BAR regions
 * ====================================================================== */

int amdgpu_lite_mmap_bar(struct amdgpu_lite_device *ldev,
			 struct vm_area_struct *vma)
{
	u64 full_offset = (u64)vma->vm_pgoff << PAGE_SHIFT;
	unsigned int bar_idx;
	resource_size_t bar_phys;
	resource_size_t bar_size;
	unsigned long size;

	/* Extract BAR index from offset encoding */
	bar_idx = (full_offset >> 40) & 0xFFFFF;
	if (bar_idx >= ldev->num_bars)
		return -EINVAL;

	bar_phys = ldev->bars[bar_idx].phys_addr;
	bar_size = ldev->bars[bar_idx].size;
	size = vma->vm_end - vma->vm_start;

	if (size > bar_size)
		return -EINVAL;

	/* Mark as IO memory: uncacheable, no swap, no merge */
	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
	vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);

	return io_remap_pfn_range(vma, vma->vm_start,
				  bar_phys >> PAGE_SHIFT,
				  size, vma->vm_page_prot);
}
