// SPDX-License-Identifier: GPL-2.0
/*
 * main.c - amdgpu_lite kernel module entry point
 *
 * Minimal kernel module providing GPU access primitives for a userspace
 * ROCm driver. Handles PCI probe/remove, char device lifecycle, and
 * dispatches ioctls to the appropriate subsystem.
 */

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/slab.h>

#include "amdgpu_lite.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("ROCm Build Infrastructure");
MODULE_DESCRIPTION("Minimal AMD GPU access for userspace driver");

/* Global device list and lock */
static LIST_HEAD(amdgpu_lite_devices);
static DEFINE_MUTEX(amdgpu_lite_devices_lock);
static int amdgpu_lite_next_index;

/* ======================================================================
 * File operations
 * ====================================================================== */

static int amdgpu_lite_open(struct inode *inode, struct file *filp)
{
	struct miscdevice *misc = filp->private_data;
	struct amdgpu_lite_device *ldev =
		container_of(misc, struct amdgpu_lite_device, misc);
	struct amdgpu_lite_fpriv *fpriv;

	fpriv = kzalloc(sizeof(*fpriv), GFP_KERNEL);
	if (!fpriv)
		return -ENOMEM;

	fpriv->ldev = ldev;
	INIT_LIST_HEAD(&fpriv->gtt_allocs);
	mutex_init(&fpriv->alloc_lock);
	fpriv->next_gtt_handle = 1;

	filp->private_data = fpriv;
	return 0;
}

static int amdgpu_lite_release(struct inode *inode, struct file *filp)
{
	struct amdgpu_lite_fpriv *fpriv = filp->private_data;

	if (fpriv) {
		/* Free all GTT allocations owned by this fd */
		amdgpu_lite_free_all_gtt(fpriv);
		mutex_destroy(&fpriv->alloc_lock);
		kfree(fpriv);
	}
	return 0;
}

static long amdgpu_lite_ioctl(struct file *filp, unsigned int cmd,
			      unsigned long arg)
{
	struct amdgpu_lite_fpriv *fpriv = filp->private_data;
	struct amdgpu_lite_device *ldev = fpriv->ldev;

	switch (cmd) {
	case AMDGPU_LITE_IOC_GET_INFO:
		return amdgpu_lite_ioctl_get_info(ldev, arg);
	case AMDGPU_LITE_IOC_MAP_BAR:
		return amdgpu_lite_ioctl_map_bar(ldev, arg);
	case AMDGPU_LITE_IOC_ALLOC_GTT:
		return amdgpu_lite_ioctl_alloc_gtt(fpriv, arg);
	case AMDGPU_LITE_IOC_FREE_GTT:
		return amdgpu_lite_ioctl_free_gtt(fpriv, arg);
	case AMDGPU_LITE_IOC_ALLOC_VRAM:
		return amdgpu_lite_ioctl_alloc_vram(fpriv, arg);
	case AMDGPU_LITE_IOC_FREE_VRAM:
		return amdgpu_lite_ioctl_free_vram(fpriv, arg);
	case AMDGPU_LITE_IOC_MAP_GPU:
		return amdgpu_lite_ioctl_map_gpu(fpriv, arg);
	case AMDGPU_LITE_IOC_UNMAP_GPU:
		return amdgpu_lite_ioctl_unmap_gpu(fpriv, arg);
	case AMDGPU_LITE_IOC_SETUP_IRQ:
		return amdgpu_lite_ioctl_setup_irq(fpriv, arg);
	default:
		return -ENOTTY;
	}
}

static int amdgpu_lite_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct amdgpu_lite_fpriv *fpriv = filp->private_data;
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	u64 mmap_type;

	mmap_type = (u64)vma->vm_pgoff << PAGE_SHIFT;
	mmap_type >>= AMDGPU_LITE_MMAP_TYPE_SHIFT;

	switch (mmap_type) {
	case AMDGPU_LITE_MMAP_TYPE_BAR:
		return amdgpu_lite_mmap_bar(ldev, vma);
	case AMDGPU_LITE_MMAP_TYPE_GTT:
		return amdgpu_lite_mmap_gtt(fpriv, vma);
	default:
		return -EINVAL;
	}
}

static const struct file_operations amdgpu_lite_fops = {
	.owner          = THIS_MODULE,
	.open           = amdgpu_lite_open,
	.release        = amdgpu_lite_release,
	.unlocked_ioctl = amdgpu_lite_ioctl,
	.mmap           = amdgpu_lite_mmap,
};

/* ======================================================================
 * PCI driver
 * ====================================================================== */

static const struct pci_device_id amdgpu_lite_pci_ids[] = {
	{ PCI_DEVICE(0x1002, 0x7551) },  /* RX 9070 XT (RDNA4 GFX1201) */
	{ 0 }
};
MODULE_DEVICE_TABLE(pci, amdgpu_lite_pci_ids);

static int amdgpu_lite_probe(struct pci_dev *pdev,
			     const struct pci_device_id *id)
{
	struct amdgpu_lite_device *ldev;
	int ret;

	dev_info(&pdev->dev, "amdgpu_lite: probing %04x:%04x\n",
		 pdev->vendor, pdev->device);

	ldev = kzalloc(sizeof(*ldev), GFP_KERNEL);
	if (!ldev)
		return -ENOMEM;

	ldev->pdev = pdev;
	INIT_LIST_HEAD(&ldev->list);
	pci_set_drvdata(pdev, ldev);

	/* Assign device index */
	mutex_lock(&amdgpu_lite_devices_lock);
	ldev->index = amdgpu_lite_next_index++;
	mutex_unlock(&amdgpu_lite_devices_lock);

	/* PCI setup: enable device, request regions, map BARs, classify */
	ret = amdgpu_lite_pci_setup(ldev);
	if (ret)
		goto err_free;

	/* Register misc device: /dev/amdgpu_lite0, /dev/amdgpu_lite1, ... */
	snprintf(ldev->misc_name, sizeof(ldev->misc_name),
		 "amdgpu_lite%d", ldev->index);
	ldev->misc.minor = MISC_DYNAMIC_MINOR;
	ldev->misc.name = ldev->misc_name;
	ldev->misc.fops = &amdgpu_lite_fops;
	ldev->misc.parent = &pdev->dev;

	ret = misc_register(&ldev->misc);
	if (ret) {
		dev_err(&pdev->dev, "amdgpu_lite: failed to register misc device\n");
		goto err_pci;
	}

	/* Add to global list */
	mutex_lock(&amdgpu_lite_devices_lock);
	list_add_tail(&ldev->list, &amdgpu_lite_devices);
	mutex_unlock(&amdgpu_lite_devices_lock);

	dev_info(&pdev->dev, "amdgpu_lite: registered as /dev/%s\n",
		 ldev->misc_name);
	return 0;

err_pci:
	amdgpu_lite_pci_cleanup(ldev);
err_free:
	kfree(ldev);
	return ret;
}

static void amdgpu_lite_remove(struct pci_dev *pdev)
{
	struct amdgpu_lite_device *ldev = pci_get_drvdata(pdev);

	if (!ldev)
		return;

	dev_info(&pdev->dev, "amdgpu_lite: removing\n");

	mutex_lock(&amdgpu_lite_devices_lock);
	list_del(&ldev->list);
	mutex_unlock(&amdgpu_lite_devices_lock);

	misc_deregister(&ldev->misc);
	amdgpu_lite_pci_cleanup(ldev);
	kfree(ldev);
}

static struct pci_driver amdgpu_lite_pci_driver = {
	.name     = AMDGPU_LITE_NAME,
	.id_table = amdgpu_lite_pci_ids,
	.probe    = amdgpu_lite_probe,
	.remove   = amdgpu_lite_remove,
};

/* ======================================================================
 * Module init/exit
 * ====================================================================== */

static int __init amdgpu_lite_init(void)
{
	pr_info("amdgpu_lite: loading\n");
	return pci_register_driver(&amdgpu_lite_pci_driver);
}

static void __exit amdgpu_lite_exit(void)
{
	pci_unregister_driver(&amdgpu_lite_pci_driver);
	pr_info("amdgpu_lite: unloaded\n");
}

module_init(amdgpu_lite_init);
module_exit(amdgpu_lite_exit);
