/*
 * ROCmGPUDriver.cpp - IOService implementation for AMD eGPU PCIe access.
 *
 * Lifecycle:
 *   1. macOS matches this driver to Thunderbolt-attached AMD PCI devices
 *   2. Start() opens the PCI device, enables bus mastering, caches BAR info
 *   3. NewUserClient() creates per-connection IOUserClient instances
 *   4. Stop() releases the PCI device
 *
 * All GPU-specific logic (IP discovery, firmware, queues) lives in Python.
 * This driver is a minimal PCIe Hardware Abstraction Layer.
 */

#include "ROCmGPUDriver.h"
#include "ROCmGPUUserClient.h"
#include "ROCmGPUShared.h"

#include <DriverKit/IOLib.h>
#include <DriverKit/IOMemoryDescriptor.iig>
#include <DriverKit/IOBufferMemoryDescriptor.iig>
#include <PCIDriverKit/IOPCIDevice.iig>

#define LOG_PREFIX "ROCmGPU: "

/* ======================================================================
 * IOService lifecycle
 * ====================================================================== */

bool ROCmGPUDriver::init()
{
    if (!super::init()) {
        return false;
    }

    fPCIDevice = nullptr;
    fBusMasterEnabled = false;

    for (int i = 0; i < 6; i++) {
        fBARs[i] = {};
    }

    IOLog(LOG_PREFIX "init()\n");
    return true;
}

kern_return_t ROCmGPUDriver::Start(IOService* provider)
{
    kern_return_t ret;

    ret = super::Start(provider);
    if (ret != kIOReturnSuccess) {
        IOLog(LOG_PREFIX "super::Start() failed: 0x%x\n", ret);
        return ret;
    }

    /* Cast provider to IOPCIDevice -- this is our PCI device handle */
    fPCIDevice = OSDynamicCast(IOPCIDevice, provider);
    if (!fPCIDevice) {
        IOLog(LOG_PREFIX "Provider is not an IOPCIDevice\n");
        return kIOReturnNoDevice;
    }

    /* Open for exclusive access -- required before any PCI operations */
    ret = fPCIDevice->Open(this, 0);
    if (ret != kIOReturnSuccess) {
        IOLog(LOG_PREFIX "Failed to open PCI device: 0x%x\n", ret);
        return ret;
    }

    /* Read vendor/device IDs to verify this is an AMD GPU */
    uint16_t vendorID = 0, deviceID = 0;
    fPCIDevice->ConfigurationRead16(0x00, &vendorID);  /* Vendor ID */
    fPCIDevice->ConfigurationRead16(0x02, &deviceID);  /* Device ID */

    IOLog(LOG_PREFIX "Matched PCI device: vendor=0x%04x device=0x%04x\n",
          vendorID, deviceID);

    if (vendorID != 0x1002) {
        IOLog(LOG_PREFIX "Not an AMD device, refusing\n");
        fPCIDevice->Close(this, 0);
        return kIOReturnUnsupported;
    }

    /* Enable bus mastering and memory space access */
    uint16_t cmdReg = 0;
    fPCIDevice->ConfigurationRead16(0x04, &cmdReg);
    cmdReg |= 0x0006;  /* BIT1 = Memory Space Enable, BIT2 = Bus Master Enable */
    fPCIDevice->ConfigurationWrite16(0x04, cmdReg);
    fBusMasterEnabled = true;

    IOLog(LOG_PREFIX "Bus mastering enabled (cmd=0x%04x)\n", cmdReg);

    /* Cache BAR information for all 6 BARs.
     *
     * DriverKit note on 64-bit BARs:
     *   A 64-bit BAR occupies two consecutive BAR registers.
     *   GetBARInfo() returns a memoryIndex that accounts for this --
     *   it's a 0-based index into the device's memory regions, not
     *   the BAR register number. For example:
     *     BAR0+BAR1 (64-bit) -> memoryIndex=0
     *     BAR2+BAR3 (64-bit) -> memoryIndex=1
     *     BAR4 (32-bit)      -> memoryIndex=2
     */
    for (uint8_t i = 0; i < 6; i++) {
        uint8_t  memIdx = 0;
        uint64_t barSize = 0;
        uint8_t  barType = 0;

        ret = fPCIDevice->GetBARInfo(i, &memIdx, &barSize, &barType);
        if (ret == kIOReturnSuccess && barSize > 0) {
            fBARs[i].memoryIndex = memIdx;
            fBARs[i].size = barSize;
            fBARs[i].type = barType;
            fBARs[i].valid = true;

            /* Determine if 64-bit by checking BAR register bits */
            uint32_t barReg = 0;
            fPCIDevice->ConfigurationRead32(0x10 + i * 4, &barReg);
            fBARs[i].is64bit = ((barReg & 0x6) == 0x4);  /* Type bits [2:1] = 10b */
            fBARs[i].prefetchable = ((barReg & 0x8) != 0);

            IOLog(LOG_PREFIX "  BAR%u: size=%lluMB memIdx=%u type=%u %s%s\n",
                  i, barSize / (1024*1024), memIdx, barType,
                  fBARs[i].is64bit ? "64-bit" : "32-bit",
                  fBARs[i].prefetchable ? " prefetchable" : "");
        } else {
            fBARs[i].valid = false;
        }
    }

    /* Register the service so IOUserClient connections can be established */
    ret = RegisterService();
    if (ret != kIOReturnSuccess) {
        IOLog(LOG_PREFIX "RegisterService() failed: 0x%x\n", ret);
        fPCIDevice->Close(this, 0);
        return ret;
    }

    IOLog(LOG_PREFIX "Start() complete — ready for user clients\n");
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUDriver::Stop(IOService* provider)
{
    IOLog(LOG_PREFIX "Stop()\n");

    if (fPCIDevice) {
        /* Disable bus mastering before releasing */
        if (fBusMasterEnabled) {
            uint16_t cmdReg = 0;
            fPCIDevice->ConfigurationRead16(0x04, &cmdReg);
            cmdReg &= ~0x0006;
            fPCIDevice->ConfigurationWrite16(0x04, cmdReg);
            fBusMasterEnabled = false;
        }

        fPCIDevice->Close(this, 0);
        fPCIDevice = nullptr;
    }

    return super::Stop(provider);
}

void ROCmGPUDriver::free()
{
    IOLog(LOG_PREFIX "free()\n");
    super::free();
}

/* ======================================================================
 * User client creation
 * ====================================================================== */

kern_return_t ROCmGPUDriver::NewUserClient(
    uint32_t type,
    IOUserClient** userClient)
{
    kern_return_t ret;
    IOService* client = nullptr;

    IOLog(LOG_PREFIX "NewUserClient(type=%u)\n", type);

    ret = Create(this, "UserClientProperties", &client);
    if (ret != kIOReturnSuccess) {
        IOLog(LOG_PREFIX "Failed to create user client: 0x%x\n", ret);
        return ret;
    }

    *userClient = OSDynamicCast(IOUserClient, client);
    if (!*userClient) {
        IOLog(LOG_PREFIX "Created service is not an IOUserClient\n");
        client->release();
        return kIOReturnError;
    }

    return kIOReturnSuccess;
}
