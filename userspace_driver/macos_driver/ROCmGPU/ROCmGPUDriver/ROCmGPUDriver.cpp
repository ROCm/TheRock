/*
 * ROCmGPUDriver.cpp - IOService implementation for AMD eGPU PCIe access.
 *
 * DriverKit naming convention:
 *   - init() / free() are virtual method overrides (direct)
 *   - Start_Impl / Stop_Impl / NewUserClient_Impl are dispatch implementations
 *     (the iig compiler generates the dispatch glue)
 */

#include <os/log.h>

#include <DriverKit/IOLib.h>
#include <DriverKit/IOMemoryDescriptor.h>
#include <DriverKit/IOBufferMemoryDescriptor.h>
#include <PCIDriverKit/IOPCIDevice.h>

/* iig-generated headers */
#include "generated/ROCmGPUDriver.h"
#include "generated/ROCmGPUUserClient.h"
#include "ROCmGPUShared.h"

#define LOG_PREFIX "ROCmGPU: "

/* ======================================================================
 * Instance variables
 * ====================================================================== */

struct ROCmGPUDriver_IVars {
    IOPCIDevice* pciDevice;
    bool         busMasterEnabled;

    struct BARInfo {
        uint64_t size;
        uint8_t  memoryIndex;
        uint8_t  type;
        bool     is64bit;
        bool     prefetchable;
        bool     valid;
    } bars[6];
};

/* ======================================================================
 * Virtual method overrides (init / free)
 * ====================================================================== */

bool ROCmGPUDriver::init()
{
    if (!super::init()) {
        return false;
    }

    ivars = IONewZero(ROCmGPUDriver_IVars, 1);
    if (!ivars) {
        return false;
    }

    os_log(OS_LOG_DEFAULT, LOG_PREFIX "init()");
    return true;
}

void ROCmGPUDriver::free()
{
    os_log(OS_LOG_DEFAULT, LOG_PREFIX "free()");
    IOSafeDeleteNULL(ivars, ROCmGPUDriver_IVars, 1);
    super::free();
}

/* ======================================================================
 * Dispatch implementations (_Impl methods)
 * ====================================================================== */

kern_return_t ROCmGPUDriver::Start_Impl(IOService* provider)
{
    kern_return_t ret;

    ret = Start(provider, SUPERDISPATCH);
    if (ret != kIOReturnSuccess) {
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "super::Start() failed: 0x%x", ret);
        return ret;
    }

    /* Cast provider to IOPCIDevice */
    ivars->pciDevice = OSDynamicCast(IOPCIDevice, provider);
    if (!ivars->pciDevice) {
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "Provider is not an IOPCIDevice");
        return kIOReturnNoDevice;
    }

    /* Open for exclusive access */
    ret = ivars->pciDevice->Open(this, 0);
    if (ret != kIOReturnSuccess) {
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "Failed to open PCI device: 0x%x", ret);
        return ret;
    }

    /* Read vendor/device IDs */
    uint16_t vendorID = 0, deviceID = 0;
    ivars->pciDevice->ConfigurationRead16(0x00, &vendorID);
    ivars->pciDevice->ConfigurationRead16(0x02, &deviceID);

    os_log(OS_LOG_DEFAULT, LOG_PREFIX "PCI device: vendor=0x%04x device=0x%04x",
           vendorID, deviceID);

    if (vendorID != 0x1002) {
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "Not an AMD device, refusing");
        ivars->pciDevice->Close(this, 0);
        return kIOReturnUnsupported;
    }

    /* Enable bus mastering and memory space */
    uint16_t cmdReg = 0;
    ivars->pciDevice->ConfigurationRead16(0x04, &cmdReg);
    cmdReg |= 0x0006;
    ivars->pciDevice->ConfigurationWrite16(0x04, cmdReg);
    ivars->busMasterEnabled = true;

    os_log(OS_LOG_DEFAULT, LOG_PREFIX "Bus mastering enabled (cmd=0x%04x)", cmdReg);

    /* Cache BAR information */
    for (uint8_t i = 0; i < 6; i++) {
        uint8_t  memIdx = 0;
        uint64_t barSize = 0;
        uint8_t  barType = 0;

        ret = ivars->pciDevice->GetBARInfo(i, &memIdx, &barSize, &barType);
        if (ret == kIOReturnSuccess && barSize > 0) {
            ivars->bars[i].memoryIndex = memIdx;
            ivars->bars[i].size = barSize;
            ivars->bars[i].type = barType;
            ivars->bars[i].valid = true;

            uint32_t barReg = 0;
            ivars->pciDevice->ConfigurationRead32(0x10 + i * 4, &barReg);
            ivars->bars[i].is64bit = ((barReg & 0x6) == 0x4);
            ivars->bars[i].prefetchable = ((barReg & 0x8) != 0);

            os_log(OS_LOG_DEFAULT, LOG_PREFIX "  BAR%u: size=%lluMB memIdx=%u type=%u %s%s",
                   i, barSize / (1024*1024), memIdx, barType,
                   ivars->bars[i].is64bit ? "64-bit" : "32-bit",
                   ivars->bars[i].prefetchable ? " prefetchable" : "");
        } else {
            ivars->bars[i].valid = false;
        }
    }

    ret = RegisterService();
    if (ret != kIOReturnSuccess) {
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "RegisterService() failed: 0x%x", ret);
        ivars->pciDevice->Close(this, 0);
        return ret;
    }

    os_log(OS_LOG_DEFAULT, LOG_PREFIX "Start() complete");
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUDriver::Stop_Impl(IOService* provider)
{
    os_log(OS_LOG_DEFAULT, LOG_PREFIX "Stop()");

    if (ivars->pciDevice) {
        if (ivars->busMasterEnabled) {
            uint16_t cmdReg = 0;
            ivars->pciDevice->ConfigurationRead16(0x04, &cmdReg);
            cmdReg &= ~0x0006;
            ivars->pciDevice->ConfigurationWrite16(0x04, cmdReg);
            ivars->busMasterEnabled = false;
        }
        ivars->pciDevice->Close(this, 0);
        ivars->pciDevice = nullptr;
    }

    return Stop(provider, SUPERDISPATCH);
}

kern_return_t ROCmGPUDriver::NewUserClient_Impl(
    uint32_t type,
    IOUserClient** userClient)
{
    kern_return_t ret;
    IOService* client = nullptr;

    os_log(OS_LOG_DEFAULT, LOG_PREFIX "NewUserClient(type=%u)", type);

    ret = Create(this, "UserClientProperties", &client);
    if (ret != kIOReturnSuccess) {
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "Failed to create user client: 0x%x", ret);
        return ret;
    }

    *userClient = OSDynamicCast(IOUserClient, client);
    if (!*userClient) {
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "Created service is not an IOUserClient");
        OSSafeReleaseNULL(client);
        return kIOReturnError;
    }

    return kIOReturnSuccess;
}
