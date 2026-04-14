/*
 * ROCmGPUDriver.h - IOService subclass for AMD eGPU PCIe access.
 *
 * This is the main DriverKit extension entry point. It matches on
 * AMD PCI devices (vendor 0x1002) attached via Thunderbolt and provides
 * low-level PCIe access (BAR mapping, MMIO, DMA, config space, interrupts)
 * to the userspace Python driver.
 *
 * Architecture:
 *   ROCmGPUDriver (IOService) -- owns the PCI device, manages lifecycle
 *   ROCmGPUUserClient (IOUserClient) -- per-client escape dispatch
 */

#ifndef ROCMGPU_DRIVER_H
#define ROCMGPU_DRIVER_H

#include <DriverKit/IOService.iig>
#include <PCIDriverKit/IOPCIDevice.iig>

/* Forward declare for the .iig -> .h generation */
class ROCmGPUUserClient;

class ROCmGPUDriver : public IOService
{
public:
    /* IOService lifecycle */
    virtual bool init() override;
    virtual kern_return_t Start(IOService* provider) override;
    virtual kern_return_t Stop(IOService* provider) override;
    virtual void free() override;

    /* Called by IOUserClient to create per-client sessions */
    virtual kern_return_t NewUserClient(
        uint32_t type,
        IOUserClient** userClient) override;

    /* ---- Accessors for IOUserClient ---- */

    IOPCIDevice* getPCIDevice() const { return fPCIDevice; }

    /* BAR info cache (populated during Start) */
    struct BARInfo {
        uint64_t size;
        uint8_t  memoryIndex;
        uint8_t  type;        /* kIOPCIResourceTypeMemory, etc. */
        bool     is64bit;
        bool     prefetchable;
        bool     valid;
    };

    const BARInfo& getBAR(unsigned index) const { return fBARs[index]; }

private:
    IOPCIDevice* fPCIDevice;
    BARInfo      fBARs[6];
    bool         fBusMasterEnabled;
};

#endif /* ROCMGPU_DRIVER_H */
