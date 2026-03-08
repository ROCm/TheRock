// SPDX-License-Identifier: GPL-2.0
/*
 * irq.c - Interrupt handling for amdgpu_lite (stub)
 *
 * When implemented: MSI-X vector allocation, ISR that signals eventfd.
 * For now, just reserves the ioctl interface.
 */

#include "amdgpu_lite.h"

long amdgpu_lite_ioctl_setup_irq(struct amdgpu_lite_fpriv *fpriv,
				 unsigned long arg)
{
	return -ENOSYS;
}
