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
 * VRAM stubs
 * ====================================================================== */

long amdgpu_lite_ioctl_alloc_vram(struct amdgpu_lite_fpriv *fpriv,
				  unsigned long arg)
{
	return -ENOSYS;
}

long amdgpu_lite_ioctl_free_vram(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg)
{
	return -ENOSYS;
}

/* ======================================================================
 * GPU page table stubs
 * ====================================================================== */

long amdgpu_lite_ioctl_map_gpu(struct amdgpu_lite_fpriv *fpriv,
			       unsigned long arg)
{
	return -ENOSYS;
}

long amdgpu_lite_ioctl_unmap_gpu(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg)
{
	return -ENOSYS;
}
