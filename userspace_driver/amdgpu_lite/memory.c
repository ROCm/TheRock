// SPDX-License-Identifier: GPL-2.0
/*
 * memory.c - Memory allocation for amdgpu_lite
 *
 * Implements GTT (DMA-coherent system memory) allocation and exposes
 * it to userspace via mmap. VRAM and GPU page table operations are
 * stubbed for future implementation.
 */

#include <linux/dma-mapping.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/mm.h>
#include <linux/vmalloc.h>
#include <linux/bitmap.h>
#include <linux/io.h>

#include "amdgpu_lite.h"

/* ======================================================================
 * GTT allocation helpers
 * ====================================================================== */

static struct amdgpu_lite_gtt_alloc *
find_gtt_by_handle(struct amdgpu_lite_fpriv *fpriv, u64 handle)
{
	struct amdgpu_lite_gtt_alloc *alloc;

	list_for_each_entry(alloc, &fpriv->gtt_allocs, list) {
		if (alloc->handle == handle)
			return alloc;
	}
	return NULL;
}

static struct amdgpu_lite_gtt_alloc *
find_gtt_by_mmap_offset(struct amdgpu_lite_fpriv *fpriv, u64 mmap_offset)
{
	struct amdgpu_lite_gtt_alloc *alloc;

	list_for_each_entry(alloc, &fpriv->gtt_allocs, list) {
		if (alloc->mmap_offset == mmap_offset)
			return alloc;
	}
	return NULL;
}

/* ======================================================================
 * ALLOC_GTT ioctl
 * ====================================================================== */

long amdgpu_lite_ioctl_alloc_gtt(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	struct amdgpu_lite_alloc_gtt params;
	struct amdgpu_lite_gtt_alloc *alloc;
	size_t size;

	if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
		return -EFAULT;

	if (params.size == 0)
		return -EINVAL;

	/* Page-align the size */
	size = PAGE_ALIGN(params.size);

	alloc = kzalloc(sizeof(*alloc), GFP_KERNEL);
	if (!alloc)
		return -ENOMEM;

	alloc->cpu_addr = dma_alloc_coherent(&ldev->pdev->dev, size,
					     &alloc->bus_addr, GFP_KERNEL);
	if (!alloc->cpu_addr) {
		kfree(alloc);
		return -ENOMEM;
	}

	alloc->size = size;

	mutex_lock(&fpriv->alloc_lock);
	alloc->handle = fpriv->next_gtt_handle++;

	/* Encode mmap offset: type=GTT, handle in middle bits */
	alloc->mmap_offset = (AMDGPU_LITE_MMAP_TYPE_GTT << AMDGPU_LITE_MMAP_TYPE_SHIFT) |
			     (alloc->handle << 40);

	list_add_tail(&alloc->list, &fpriv->gtt_allocs);
	mutex_unlock(&fpriv->alloc_lock);

	/* Fill output */
	params.handle = alloc->handle;
	params.bus_addr = alloc->bus_addr;
	params.mmap_offset = alloc->mmap_offset;

	if (copy_to_user((void __user *)arg, &params, sizeof(params)))
		return -EFAULT;

	dev_dbg(&ldev->pdev->dev,
		"amdgpu_lite: alloc_gtt handle=%llu size=%zu bus=0x%llx\n",
		alloc->handle, size, (u64)alloc->bus_addr);

	return 0;
}

/* ======================================================================
 * FREE_GTT ioctl
 * ====================================================================== */

long amdgpu_lite_ioctl_free_gtt(struct amdgpu_lite_fpriv *fpriv,
				unsigned long arg)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	struct amdgpu_lite_free_gtt params;
	struct amdgpu_lite_gtt_alloc *alloc;

	if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
		return -EFAULT;

	mutex_lock(&fpriv->alloc_lock);
	alloc = find_gtt_by_handle(fpriv, params.handle);
	if (!alloc) {
		mutex_unlock(&fpriv->alloc_lock);
		return -ENOENT;
	}

	list_del(&alloc->list);
	mutex_unlock(&fpriv->alloc_lock);

	dma_free_coherent(&ldev->pdev->dev, alloc->size,
			  alloc->cpu_addr, alloc->bus_addr);
	kfree(alloc);

	return 0;
}

/* ======================================================================
 * Free all GTT allocations (called on file close)
 * ====================================================================== */

void amdgpu_lite_free_all_gtt(struct amdgpu_lite_fpriv *fpriv)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	struct amdgpu_lite_gtt_alloc *alloc, *tmp;

	mutex_lock(&fpriv->alloc_lock);
	list_for_each_entry_safe(alloc, tmp, &fpriv->gtt_allocs, list) {
		list_del(&alloc->list);
		dma_free_coherent(&ldev->pdev->dev, alloc->size,
				  alloc->cpu_addr, alloc->bus_addr);
		kfree(alloc);
	}
	mutex_unlock(&fpriv->alloc_lock);
}

/* ======================================================================
 * mmap handler for GTT regions
 * ====================================================================== */

int amdgpu_lite_mmap_gtt(struct amdgpu_lite_fpriv *fpriv,
			 struct vm_area_struct *vma)
{
	u64 full_offset = (u64)vma->vm_pgoff << PAGE_SHIFT;
	struct amdgpu_lite_gtt_alloc *alloc;
	unsigned long size = vma->vm_end - vma->vm_start;
	unsigned long pfn;

	mutex_lock(&fpriv->alloc_lock);
	alloc = find_gtt_by_mmap_offset(fpriv, full_offset);
	if (!alloc) {
		mutex_unlock(&fpriv->alloc_lock);
		return -EINVAL;
	}

	if (size > alloc->size) {
		mutex_unlock(&fpriv->alloc_lock);
		return -EINVAL;
	}

	pfn = virt_to_phys(alloc->cpu_addr) >> PAGE_SHIFT;
	mutex_unlock(&fpriv->alloc_lock);

	/* DMA-coherent memory: use noncached mapping */
	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
	vm_flags_set(vma, VM_DONTEXPAND | VM_DONTDUMP);

	return remap_pfn_range(vma, vma->vm_start, pfn, size,
			       vma->vm_page_prot);
}

/* ======================================================================
 * VRAM allocator init/destroy
 * ====================================================================== */

int vram_allocator_init(struct amdgpu_lite_device *ldev)
{
	struct vram_allocator *va = &ldev->vram;
	u64 vram_size;

	/*
	 * Use visible_vram_size since that's what we can map through the BAR.
	 * If visible_vram_size is 0 (no VRAM detected), skip init.
	 */
	vram_size = ldev->visible_vram_size;
	if (vram_size == 0) {
		dev_warn(&ldev->pdev->dev,
			 "amdgpu_lite: no visible VRAM, allocator disabled\n");
		va->bitmap = NULL;
		va->total_pages = 0;
		va->free_pages = 0;
		mutex_init(&va->lock);
		return 0;
	}

	va->total_pages = vram_size >> PAGE_SHIFT;
	va->free_pages = va->total_pages;
	mutex_init(&va->lock);

	va->bitmap = vzalloc(BITS_TO_LONGS(va->total_pages) * sizeof(unsigned long));
	if (!va->bitmap) {
		dev_err(&ldev->pdev->dev,
			"amdgpu_lite: failed to allocate VRAM bitmap (%llu pages)\n",
			va->total_pages);
		return -ENOMEM;
	}

	dev_info(&ldev->pdev->dev,
		 "amdgpu_lite: VRAM allocator initialized: %llu pages (%llu MB)\n",
		 va->total_pages, vram_size / (1024 * 1024));

	return 0;
}

void vram_allocator_destroy(struct amdgpu_lite_device *ldev)
{
	struct vram_allocator *va = &ldev->vram;

	mutex_destroy(&va->lock);
	vfree(va->bitmap);
	va->bitmap = NULL;
}

/* ======================================================================
 * VRAM allocation helpers
 * ====================================================================== */

static struct vram_allocation *
find_vram_by_handle(struct amdgpu_lite_fpriv *fpriv, u64 handle)
{
	struct vram_allocation *alloc;

	list_for_each_entry(alloc, &fpriv->vram_allocs, list) {
		if (alloc->handle == handle)
			return alloc;
	}
	return NULL;
}

static struct vram_allocation *
find_vram_by_mmap_offset(struct amdgpu_lite_fpriv *fpriv, u64 mmap_offset)
{
	struct vram_allocation *alloc;

	list_for_each_entry(alloc, &fpriv->vram_allocs, list) {
		u64 expected = (AMDGPU_LITE_MMAP_TYPE_VRAM << AMDGPU_LITE_MMAP_TYPE_SHIFT) |
			       alloc->gpu_offset;
		if (expected == mmap_offset)
			return alloc;
	}
	return NULL;
}

/* ======================================================================
 * ALLOC_VRAM ioctl
 * ====================================================================== */

long amdgpu_lite_ioctl_alloc_vram(struct amdgpu_lite_fpriv *fpriv,
				  unsigned long arg)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	struct vram_allocator *va = &ldev->vram;
	struct amdgpu_lite_alloc_vram params;
	struct vram_allocation *alloc;
	u32 num_pages;
	unsigned long page_idx;

	if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
		return -EFAULT;

	if (params.size == 0)
		return -EINVAL;

	if (!va->bitmap)
		return -ENOMEM;

	num_pages = (PAGE_ALIGN(params.size)) >> PAGE_SHIFT;

	alloc = kzalloc(sizeof(*alloc), GFP_KERNEL);
	if (!alloc)
		return -ENOMEM;

	mutex_lock(&va->lock);

	if (num_pages > va->free_pages) {
		mutex_unlock(&va->lock);
		kfree(alloc);
		return -ENOMEM;
	}

	/* Find contiguous free pages in the bitmap */
	page_idx = bitmap_find_next_zero_area(va->bitmap, va->total_pages,
					      0, num_pages, 0);
	if (page_idx >= va->total_pages) {
		mutex_unlock(&va->lock);
		kfree(alloc);
		return -ENOMEM;
	}

	/* Mark pages as allocated */
	bitmap_set(va->bitmap, page_idx, num_pages);
	va->free_pages -= num_pages;

	mutex_unlock(&va->lock);

	alloc->gpu_offset = (u64)page_idx << PAGE_SHIFT;
	alloc->size = (u64)num_pages << PAGE_SHIFT;
	alloc->num_pages = num_pages;

	mutex_lock(&fpriv->alloc_lock);
	alloc->handle = fpriv->next_vram_handle++;
	list_add_tail(&alloc->list, &fpriv->vram_allocs);
	mutex_unlock(&fpriv->alloc_lock);

	/* Fill output */
	params.handle = alloc->handle;
	params.gpu_addr = ldev->bars[ldev->vram_bar_idx].phys_addr + alloc->gpu_offset;
	params.mmap_offset = (AMDGPU_LITE_MMAP_TYPE_VRAM << AMDGPU_LITE_MMAP_TYPE_SHIFT) |
			     alloc->gpu_offset;

	if (copy_to_user((void __user *)arg, &params, sizeof(params))) {
		/* Rollback on failure */
		mutex_lock(&fpriv->alloc_lock);
		list_del(&alloc->list);
		mutex_unlock(&fpriv->alloc_lock);
		mutex_lock(&va->lock);
		bitmap_clear(va->bitmap, page_idx, num_pages);
		va->free_pages += num_pages;
		mutex_unlock(&va->lock);
		kfree(alloc);
		return -EFAULT;
	}

	dev_dbg(&ldev->pdev->dev,
		"amdgpu_lite: alloc_vram handle=%llu size=%llu offset=0x%llx\n",
		alloc->handle, alloc->size, alloc->gpu_offset);

	return 0;
}

/* ======================================================================
 * FREE_VRAM ioctl
 * ====================================================================== */

long amdgpu_lite_ioctl_free_vram(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	struct vram_allocator *va = &ldev->vram;
	struct amdgpu_lite_free_vram params;
	struct vram_allocation *alloc;
	unsigned long page_idx;

	if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
		return -EFAULT;

	mutex_lock(&fpriv->alloc_lock);
	alloc = find_vram_by_handle(fpriv, params.handle);
	if (!alloc) {
		mutex_unlock(&fpriv->alloc_lock);
		return -ENOENT;
	}

	list_del(&alloc->list);
	mutex_unlock(&fpriv->alloc_lock);

	/* Free pages in the bitmap */
	page_idx = alloc->gpu_offset >> PAGE_SHIFT;

	mutex_lock(&va->lock);
	bitmap_clear(va->bitmap, page_idx, alloc->num_pages);
	va->free_pages += alloc->num_pages;
	mutex_unlock(&va->lock);

	dev_dbg(&ldev->pdev->dev,
		"amdgpu_lite: free_vram handle=%llu offset=0x%llx\n",
		alloc->handle, alloc->gpu_offset);

	kfree(alloc);
	return 0;
}

/* ======================================================================
 * Free all VRAM allocations (called on file close)
 * ====================================================================== */

void amdgpu_lite_free_all_vram(struct amdgpu_lite_fpriv *fpriv)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	struct vram_allocator *va = &ldev->vram;
	struct vram_allocation *alloc, *tmp;

	mutex_lock(&fpriv->alloc_lock);
	list_for_each_entry_safe(alloc, tmp, &fpriv->vram_allocs, list) {
		unsigned long page_idx = alloc->gpu_offset >> PAGE_SHIFT;

		list_del(&alloc->list);

		mutex_lock(&va->lock);
		bitmap_clear(va->bitmap, page_idx, alloc->num_pages);
		va->free_pages += alloc->num_pages;
		mutex_unlock(&va->lock);

		kfree(alloc);
	}
	mutex_unlock(&fpriv->alloc_lock);
}

/* ======================================================================
 * mmap handler for VRAM regions
 * ====================================================================== */

int amdgpu_lite_mmap_vram(struct amdgpu_lite_fpriv *fpriv,
			  struct vm_area_struct *vma)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	u64 full_offset = (u64)vma->vm_pgoff << PAGE_SHIFT;
	struct vram_allocation *alloc;
	unsigned long size = vma->vm_end - vma->vm_start;
	resource_size_t bar_phys;
	u64 alloc_offset;

	mutex_lock(&fpriv->alloc_lock);
	alloc = find_vram_by_mmap_offset(fpriv, full_offset);
	if (!alloc) {
		mutex_unlock(&fpriv->alloc_lock);
		return -EINVAL;
	}

	if (size > alloc->size) {
		mutex_unlock(&fpriv->alloc_lock);
		return -EINVAL;
	}

	alloc_offset = alloc->gpu_offset;
	mutex_unlock(&fpriv->alloc_lock);

	/* Map through the VRAM BAR at the allocation's offset */
	bar_phys = ldev->bars[ldev->vram_bar_idx].phys_addr + alloc_offset;

	/* Use write-combining for VRAM (better performance than uncached) */
	vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
	vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);

	return io_remap_pfn_range(vma, vma->vm_start,
				  bar_phys >> PAGE_SHIFT,
				  size, vma->vm_page_prot);
}

/* ======================================================================
 * GART page table - PTE format (GFX12)
 * ====================================================================== */

#define AMDGPU_PTE_VALID      (1ULL << 0)
#define AMDGPU_PTE_SYSTEM     (1ULL << 1)
#define AMDGPU_PTE_SNOOPED    (1ULL << 2)
#define AMDGPU_PTE_EXECUTABLE (1ULL << 4)
#define AMDGPU_PTE_READABLE   (1ULL << 5)
#define AMDGPU_PTE_WRITEABLE  (1ULL << 6)
#define AMDGPU_PTE_IS_PTE     (1ULL << 63)
#define AMDGPU_PTE_MTYPE_UC   (3ULL << 54)

static uint64_t build_gart_pte(dma_addr_t bus_addr)
{
	uint64_t pte = AMDGPU_PTE_VALID | AMDGPU_PTE_SYSTEM | AMDGPU_PTE_SNOOPED |
		       AMDGPU_PTE_READABLE | AMDGPU_PTE_WRITEABLE |
		       AMDGPU_PTE_EXECUTABLE |
		       AMDGPU_PTE_IS_PTE | AMDGPU_PTE_MTYPE_UC;
	pte |= (bus_addr & 0x0000FFFFF000ULL);  /* Bits [47:12] */
	return pte;
}

/* ======================================================================
 * GART table init / cleanup
 * ====================================================================== */

int amdgpu_lite_gart_init(struct amdgpu_lite_device *ldev)
{
	mutex_init(&ldev->gart_lock);
	ldev->gart_size = AMDGPU_LITE_GART_TABLE_SIZE;
	ldev->gart_next_gpu_va = AMDGPU_LITE_GART_VA_START;

	ldev->gart_table = dma_alloc_coherent(&ldev->pdev->dev,
					      ldev->gart_size,
					      &ldev->gart_table_bus_addr,
					      GFP_KERNEL);
	if (!ldev->gart_table) {
		dev_err(&ldev->pdev->dev,
			"amdgpu_lite: failed to allocate GART table (%llu bytes)\n",
			ldev->gart_size);
		return -ENOMEM;
	}

	/* Zero out all PTEs */
	memset(ldev->gart_table, 0, ldev->gart_size);

	dev_info(&ldev->pdev->dev,
		 "amdgpu_lite: GART table allocated: %llu entries, bus=0x%llx, VA start=0x%llx\n",
		 (u64)AMDGPU_LITE_GART_NUM_ENTRIES,
		 (u64)ldev->gart_table_bus_addr,
		 ldev->gart_next_gpu_va);

	return 0;
}

void amdgpu_lite_gart_cleanup(struct amdgpu_lite_device *ldev)
{
	if (ldev->gart_table) {
		dma_free_coherent(&ldev->pdev->dev, ldev->gart_size,
				  ldev->gart_table, ldev->gart_table_bus_addr);
		ldev->gart_table = NULL;
	}
	mutex_destroy(&ldev->gart_lock);
}

/* ======================================================================
 * MAP_GPU ioctl - write PTEs into GART table
 * ====================================================================== */

long amdgpu_lite_ioctl_map_gpu(struct amdgpu_lite_fpriv *fpriv,
			       unsigned long arg)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	struct amdgpu_lite_map_gpu params;
	struct amdgpu_lite_gtt_alloc *alloc;
	u64 gpu_va, size, page_index;
	u64 num_pages, i;

	if (!ldev->gart_table)
		return -ENODEV;

	if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
		return -EFAULT;

	if (params.size == 0)
		return -EINVAL;

	size = PAGE_ALIGN(params.size);
	num_pages = size >> PAGE_SHIFT;

	/* Look up the GTT allocation by handle to get bus_addr */
	mutex_lock(&fpriv->alloc_lock);
	alloc = find_gtt_by_handle(fpriv, params.handle);
	if (!alloc) {
		mutex_unlock(&fpriv->alloc_lock);
		return -ENOENT;
	}

	if (size > alloc->size) {
		mutex_unlock(&fpriv->alloc_lock);
		return -EINVAL;
	}

	mutex_lock(&ldev->gart_lock);

	/* Determine GPU VA: auto-assign or use requested */
	if (params.gpu_va == 0) {
		gpu_va = ldev->gart_next_gpu_va;
	} else {
		gpu_va = params.gpu_va;
		if (gpu_va & (PAGE_SIZE - 1)) {
			mutex_unlock(&ldev->gart_lock);
			mutex_unlock(&fpriv->alloc_lock);
			return -EINVAL;
		}
	}

	/* Calculate page index relative to GART VA start */
	if (gpu_va < AMDGPU_LITE_GART_VA_START) {
		mutex_unlock(&ldev->gart_lock);
		mutex_unlock(&fpriv->alloc_lock);
		return -EINVAL;
	}

	page_index = (gpu_va - AMDGPU_LITE_GART_VA_START) >> PAGE_SHIFT;

	/* Check bounds */
	if (page_index + num_pages > AMDGPU_LITE_GART_NUM_ENTRIES) {
		mutex_unlock(&ldev->gart_lock);
		mutex_unlock(&fpriv->alloc_lock);
		return -ENOSPC;
	}

	/* Write PTEs */
	for (i = 0; i < num_pages; i++) {
		ldev->gart_table[page_index + i] =
			build_gart_pte(alloc->bus_addr + i * PAGE_SIZE);
	}

	/* Advance bump allocator if we used auto-assign */
	if (params.gpu_va == 0) {
		ldev->gart_next_gpu_va = gpu_va + size;
	}

	mutex_unlock(&ldev->gart_lock);
	mutex_unlock(&fpriv->alloc_lock);

	/* Return assigned GPU VA */
	params.mapped_gpu_va = gpu_va;

	if (copy_to_user((void __user *)arg, &params, sizeof(params)))
		return -EFAULT;

	dev_dbg(&ldev->pdev->dev,
		"amdgpu_lite: map_gpu handle=%llu gpu_va=0x%llx pages=%llu\n",
		params.handle, gpu_va, num_pages);

	return 0;
}

/* ======================================================================
 * UNMAP_GPU ioctl - clear PTEs in GART table
 * ====================================================================== */

long amdgpu_lite_ioctl_unmap_gpu(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	struct amdgpu_lite_unmap_gpu params;
	u64 page_index, num_pages, i;

	if (!ldev->gart_table)
		return -ENODEV;

	if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
		return -EFAULT;

	if (params.size == 0 || (params.gpu_va & (PAGE_SIZE - 1)))
		return -EINVAL;

	num_pages = PAGE_ALIGN(params.size) >> PAGE_SHIFT;

	if (params.gpu_va < AMDGPU_LITE_GART_VA_START)
		return -EINVAL;

	page_index = (params.gpu_va - AMDGPU_LITE_GART_VA_START) >> PAGE_SHIFT;

	if (page_index + num_pages > AMDGPU_LITE_GART_NUM_ENTRIES)
		return -EINVAL;

	mutex_lock(&ldev->gart_lock);
	for (i = 0; i < num_pages; i++)
		ldev->gart_table[page_index + i] = 0;
	mutex_unlock(&ldev->gart_lock);

	dev_dbg(&ldev->pdev->dev,
		"amdgpu_lite: unmap_gpu gpu_va=0x%llx pages=%llu\n",
		params.gpu_va, num_pages);

	return 0;
}
