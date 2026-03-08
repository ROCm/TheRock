// SPDX-License-Identifier: GPL-2.0
/*
 * irq.c - MSI-X interrupt handling and eventfd forwarding for amdgpu_lite
 *
 * Allocates MSI/MSI-X vectors, registers an ISR that signals all registered
 * eventfds, and provides SETUP_IRQ ioctl for userspace to register/teardown
 * eventfd notifications.
 */

#include <linux/pci.h>
#include <linux/interrupt.h>
#include <linux/eventfd.h>
#include <linux/file.h>
#include <linux/uaccess.h>

#include "amdgpu_lite.h"

/* ======================================================================
 * ISR - Interrupt service routine
 *
 * For MSI/MSI-X, the interrupt is guaranteed to be ours, so we always
 * claim it. Signal all registered eventfds.
 * ====================================================================== */

static irqreturn_t amdgpu_lite_irq_handler(int irq, void *data)
{
	struct amdgpu_lite_device *ldev = data;
	unsigned long flags;
	int i;

	spin_lock_irqsave(&ldev->events_lock, flags);
	for (i = 0; i < AMDGPU_LITE_MAX_EVENTS; i++) {
		if (ldev->events[i].in_use && ldev->events[i].ctx)
			eventfd_signal(ldev->events[i].ctx);
	}
	spin_unlock_irqrestore(&ldev->events_lock, flags);

	return IRQ_HANDLED;
}

/* ======================================================================
 * IRQ init / cleanup (called from probe/remove)
 * ====================================================================== */

int amdgpu_lite_irq_init(struct amdgpu_lite_device *ldev)
{
	struct pci_dev *pdev = ldev->pdev;
	int nvecs, ret;

	spin_lock_init(&ldev->events_lock);
	memset(ldev->events, 0, sizeof(ldev->events));
	ldev->irq_vector = -1;

	/* Try MSI-X first, fall back to MSI */
	nvecs = pci_alloc_irq_vectors(pdev, 1, 1,
				      PCI_IRQ_MSIX | PCI_IRQ_MSI);
	if (nvecs < 0) {
		dev_warn(&pdev->dev,
			 "amdgpu_lite: failed to allocate MSI-X/MSI vectors: %d\n",
			 nvecs);
		return 0;  /* Non-fatal: IRQ just won't work */
	}

	ldev->irq_vector = pci_irq_vector(pdev, 0);
	if (ldev->irq_vector < 0) {
		dev_warn(&pdev->dev,
			 "amdgpu_lite: pci_irq_vector failed: %d\n",
			 ldev->irq_vector);
		pci_free_irq_vectors(pdev);
		ldev->irq_vector = -1;
		return 0;
	}

	ret = request_irq(ldev->irq_vector, amdgpu_lite_irq_handler,
			  0, "amdgpu_lite", ldev);
	if (ret) {
		dev_warn(&pdev->dev,
			 "amdgpu_lite: request_irq failed: %d\n", ret);
		pci_free_irq_vectors(pdev);
		ldev->irq_vector = -1;
		return 0;
	}

	dev_info(&pdev->dev,
		 "amdgpu_lite: IRQ vector %d allocated\n", ldev->irq_vector);
	return 0;
}

void amdgpu_lite_irq_cleanup(struct amdgpu_lite_device *ldev)
{
	unsigned long flags;
	int i;

	/* Free the IRQ and vectors */
	if (ldev->irq_vector >= 0) {
		free_irq(ldev->irq_vector, ldev);
		pci_free_irq_vectors(ldev->pdev);
		ldev->irq_vector = -1;
	}

	/* Release all eventfd contexts */
	spin_lock_irqsave(&ldev->events_lock, flags);
	for (i = 0; i < AMDGPU_LITE_MAX_EVENTS; i++) {
		if (ldev->events[i].in_use && ldev->events[i].ctx) {
			eventfd_ctx_put(ldev->events[i].ctx);
			ldev->events[i].ctx = NULL;
			ldev->events[i].in_use = false;
			ldev->events[i].owner = NULL;
		}
	}
	spin_unlock_irqrestore(&ldev->events_lock, flags);
}

/* ======================================================================
 * Release all eventfd registrations owned by a file (called on close)
 * ====================================================================== */

void amdgpu_lite_irq_release_fpriv(struct amdgpu_lite_fpriv *fpriv)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	unsigned long flags;
	int i;

	spin_lock_irqsave(&ldev->events_lock, flags);
	for (i = 0; i < AMDGPU_LITE_MAX_EVENTS; i++) {
		if (ldev->events[i].in_use && ldev->events[i].owner == fpriv) {
			if (ldev->events[i].ctx)
				eventfd_ctx_put(ldev->events[i].ctx);
			ldev->events[i].ctx = NULL;
			ldev->events[i].in_use = false;
			ldev->events[i].owner = NULL;
		}
	}
	spin_unlock_irqrestore(&ldev->events_lock, flags);
}

/* ======================================================================
 * SETUP_IRQ ioctl
 *
 * If registration_id == 0: register a new eventfd
 * If registration_id != 0: teardown that registration
 * ====================================================================== */

long amdgpu_lite_ioctl_setup_irq(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg)
{
	struct amdgpu_lite_device *ldev = fpriv->ldev;
	struct amdgpu_lite_setup_irq params;
	struct eventfd_ctx *ctx;
	unsigned long flags;
	int i;

	if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
		return -EFAULT;

	/* Teardown path */
	if (params.registration_id != 0) {
		uint32_t reg_id = params.registration_id;

		if (reg_id > AMDGPU_LITE_MAX_EVENTS)
			return -EINVAL;

		i = reg_id - 1;  /* registration_id is 1-based */

		spin_lock_irqsave(&ldev->events_lock, flags);
		if (!ldev->events[i].in_use ||
		    ldev->events[i].owner != fpriv) {
			spin_unlock_irqrestore(&ldev->events_lock, flags);
			return -ENOENT;
		}

		if (ldev->events[i].ctx)
			eventfd_ctx_put(ldev->events[i].ctx);
		ldev->events[i].ctx = NULL;
		ldev->events[i].in_use = false;
		ldev->events[i].owner = NULL;
		spin_unlock_irqrestore(&ldev->events_lock, flags);

		return 0;
	}

	/* Registration path */
	if (ldev->irq_vector < 0)
		return -ENODEV;  /* No IRQ available */

	ctx = eventfd_ctx_fdget(params.eventfd);
	if (IS_ERR(ctx))
		return PTR_ERR(ctx);

	spin_lock_irqsave(&ldev->events_lock, flags);

	/* Find a free slot */
	for (i = 0; i < AMDGPU_LITE_MAX_EVENTS; i++) {
		if (!ldev->events[i].in_use)
			break;
	}

	if (i >= AMDGPU_LITE_MAX_EVENTS) {
		spin_unlock_irqrestore(&ldev->events_lock, flags);
		eventfd_ctx_put(ctx);
		return -ENOSPC;
	}

	ldev->events[i].ctx = ctx;
	ldev->events[i].irq_source = params.irq_source;
	ldev->events[i].owner = fpriv;
	ldev->events[i].in_use = true;

	spin_unlock_irqrestore(&ldev->events_lock, flags);

	/* Return 1-based registration ID */
	params.out_registration_id = i + 1;

	if (copy_to_user((void __user *)arg, &params, sizeof(params))) {
		/* Undo registration on copy failure */
		spin_lock_irqsave(&ldev->events_lock, flags);
		ldev->events[i].in_use = false;
		ldev->events[i].owner = NULL;
		spin_unlock_irqrestore(&ldev->events_lock, flags);
		eventfd_ctx_put(ctx);
		return -EFAULT;
	}

	dev_dbg(&ldev->pdev->dev,
		"amdgpu_lite: registered eventfd irq_source=%u reg_id=%u\n",
		params.irq_source, i + 1);

	return 0;
}
