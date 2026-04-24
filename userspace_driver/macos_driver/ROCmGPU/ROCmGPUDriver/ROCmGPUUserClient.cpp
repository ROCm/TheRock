/*
 * ROCmGPUUserClient.cpp - Escape command dispatch implementation.
 *
 * DriverKit conventions used:
 *   - _Impl suffix for dispatch methods (Start_Impl, Stop_Impl, etc.)
 *   - VirtualMethods (ExternalMethod, init, free) are direct overrides
 *   - Handler functions are file-static (not class members, since
 *     iig-generated classes only expose _Impl and VirtualMethods)
 *   - structureOutput is OSData* (created with OSData::withBytes)
 *   - ivars pattern for instance variables
 */

#include <os/log.h>
#include <string.h>

#include <DriverKit/IOLib.h>
#include <DriverKit/IOUserClient.h>
#include <DriverKit/IOBufferMemoryDescriptor.h>
#include <DriverKit/IODMACommand.h>
#include <DriverKit/IOInterruptDispatchSource.h>
#include <DriverKit/IODispatchQueue.h>
#include <DriverKit/OSData.h>
#include <PCIDriverKit/IOPCIDevice.h>

#include "generated/ROCmGPUUserClient.h"
#include "generated/ROCmGPUDriver.h"
#include "ROCmGPUShared.h"

#define LOG_PREFIX "ROCmGPU-UC: "
#define MAX_DMA_BUFFERS 256

/* ======================================================================
 * Instance variables
 * ====================================================================== */

struct ROCmGPUUserClient_IVars {
    ROCmGPUDriver* driver;
    IOPCIDevice*   pciDevice;

    struct DMABuffer {
        IOBufferMemoryDescriptor* descriptor;
        IODMACommand*             dmaCommand;
        uint64_t                  physAddr;
        uint64_t                  size;
        bool                      inUse;
    };

    DMABuffer dmaBuffers[MAX_DMA_BUFFERS];
    uint32_t  nextDMAID;

    IOInterruptDispatchSource* intSource;
    bool intFired;
};

/* Helper: get BAR memoryIndex from PCI device */
static kern_return_t getBarMemoryIndex(IOPCIDevice* dev, uint8_t barIndex,
                                       uint8_t* memIdx, uint64_t* barSize)
{
    uint8_t type = 0;
    return dev->GetBARInfo(barIndex, memIdx, barSize, &type);
}

/* ======================================================================
 * Dispatch table
 * ====================================================================== */

static const IOUserClientMethodDispatch sMethods[kROCmGPU_SelectorCount] = {
    [kROCmGPU_GetInfo]       = { nullptr, false, 0, 0, 0, sizeof(ROCmGPUDeviceInfo) },
    [kROCmGPU_Reset]         = { nullptr, false, 0, 0, 0, 0 },
    [kROCmGPU_CfgRead]       = { nullptr, false, 2, 1, 0, 0 },
    [kROCmGPU_CfgWrite]      = { nullptr, false, 3, 0, 0, 0 },
    [kROCmGPU_MMIORead32]    = { nullptr, false, 2, 1, 0, 0 },
    [kROCmGPU_MMIOWrite32]   = { nullptr, false, 3, 0, 0, 0 },
    [kROCmGPU_MapBAR]        = { nullptr, false, 1, 1, 0, 0 },
    [kROCmGPU_UnmapBAR]      = { nullptr, false, 1, 0, 0, 0 },
    [kROCmGPU_AllocDMA]      = { nullptr, false, 2, 0, 0, sizeof(ROCmGPUDMAInfo) },
    [kROCmGPU_FreeDMA]       = { nullptr, false, 1, 0, 0, 0 },
    [kROCmGPU_MapDMA]        = { nullptr, false, 1, 1, 0, 0 },
    [kROCmGPU_EnableMSI]     = { nullptr, false, 1, 0, 0, 0 },
    [kROCmGPU_WaitInterrupt] = { nullptr, false, 1, 1, 0, 0 },
};

/* ======================================================================
 * Static handler functions (called from ExternalMethod dispatch)
 * ====================================================================== */

static kern_return_t handleGetInfo(ROCmGPUUserClient_IVars* iv,
                                   IOUserClientMethodArguments* args)
{
    ROCmGPUDeviceInfo info = {};
    auto* dev = iv->pciDevice;

    dev->ConfigurationRead16(0x00, &info.vendorID);
    dev->ConfigurationRead16(0x02, &info.deviceID);
    dev->ConfigurationRead16(0x2C, &info.subsystemVendorID);
    dev->ConfigurationRead16(0x2E, &info.subsystemDeviceID);
    uint8_t rev = 0;
    dev->ConfigurationRead8(0x08, &rev);
    info.revisionID = rev;

    uint64_t maxVRAM = 0;
    for (int i = 0; i < 6; i++) {
        uint8_t memIdx = 0; uint64_t barSz = 0; uint8_t barTy = 0;
        if (dev->GetBARInfo(i, &memIdx, &barSz, &barTy) == kIOReturnSuccess) {
            info.bars[i].size = barSz;
            info.bars[i].memoryIndex = memIdx;
            info.bars[i].type = barTy;
            uint32_t barReg = 0;
            dev->ConfigurationRead32(0x10 + i * 4, &barReg);
            info.bars[i].is64bit = ((barReg & 0x6) == 0x4) ? 1 : 0;
            info.bars[i].prefetchable = ((barReg & 0x8) != 0) ? 1 : 0;
            if (info.bars[i].prefetchable && barSz > maxVRAM) maxVRAM = barSz;
        }
    }
    info.vramSize = maxVRAM;

    args->structureOutput = OSData::withBytes(&info, sizeof(info));
    return args->structureOutput ? kIOReturnSuccess : kIOReturnNoMemory;
}

static kern_return_t handleReset(ROCmGPUUserClient_IVars* iv,
                                 IOUserClientMethodArguments* args)
{
    uint64_t cap = 0;
    if (iv->pciDevice->FindPCICapability(0x10, 0, &cap) != kIOReturnSuccess)
        return kIOReturnUnsupported;

    uint16_t devCtl = 0;
    iv->pciDevice->ConfigurationRead16(cap + 0x08, &devCtl);
    devCtl |= (1 << 15);
    iv->pciDevice->ConfigurationWrite16(cap + 0x08, devCtl);
    return kIOReturnSuccess;
}

static kern_return_t handleCfgRead(ROCmGPUUserClient_IVars* iv,
                                   IOUserClientMethodArguments* args)
{
    uint64_t offset = args->scalarInput[0];
    uint64_t width  = args->scalarInput[1];
    uint64_t value  = 0;

    switch (width) {
    case 1: { uint8_t v;  iv->pciDevice->ConfigurationRead8(offset, &v);  value = v; break; }
    case 2: { uint16_t v; iv->pciDevice->ConfigurationRead16(offset, &v); value = v; break; }
    case 4: { uint32_t v; iv->pciDevice->ConfigurationRead32(offset, &v); value = v; break; }
    default: return kIOReturnBadArgument;
    }
    args->scalarOutput[0] = value;
    return kIOReturnSuccess;
}

static kern_return_t handleCfgWrite(ROCmGPUUserClient_IVars* iv,
                                    IOUserClientMethodArguments* args)
{
    uint64_t offset = args->scalarInput[0];
    uint64_t width  = args->scalarInput[1];
    uint64_t value  = args->scalarInput[2];

    switch (width) {
    case 1: iv->pciDevice->ConfigurationWrite8(offset, (uint8_t)value);   break;
    case 2: iv->pciDevice->ConfigurationWrite16(offset, (uint16_t)value); break;
    case 4: iv->pciDevice->ConfigurationWrite32(offset, (uint32_t)value); break;
    default: return kIOReturnBadArgument;
    }
    return kIOReturnSuccess;
}

static kern_return_t handleMMIORead32(ROCmGPUUserClient_IVars* iv,
                                      IOUserClientMethodArguments* args)
{
    uint8_t barIdx = (uint8_t)args->scalarInput[0];
    uint64_t offset = args->scalarInput[1];
    uint8_t memIdx = 0; uint64_t barSz = 0;
    if (getBarMemoryIndex(iv->pciDevice, barIdx, &memIdx, &barSz) != kIOReturnSuccess)
        return kIOReturnNotFound;
    if (offset + 4 > barSz) return kIOReturnBadArgument;

    uint32_t value = 0;
    iv->pciDevice->MemoryRead32(memIdx, offset, &value);
    args->scalarOutput[0] = value;
    return kIOReturnSuccess;
}

static kern_return_t handleMMIOWrite32(ROCmGPUUserClient_IVars* iv,
                                       IOUserClientMethodArguments* args)
{
    uint8_t barIdx = (uint8_t)args->scalarInput[0];
    uint64_t offset = args->scalarInput[1];
    uint32_t value  = (uint32_t)args->scalarInput[2];
    uint8_t memIdx = 0; uint64_t barSz = 0;
    if (getBarMemoryIndex(iv->pciDevice, barIdx, &memIdx, &barSz) != kIOReturnSuccess)
        return kIOReturnNotFound;
    if (offset + 4 > barSz) return kIOReturnBadArgument;

    iv->pciDevice->MemoryWrite32(memIdx, offset, value);
    return kIOReturnSuccess;
}

static kern_return_t handleMapBAR(ROCmGPUUserClient_IVars* iv,
                                  IOUserClientMethodArguments* args)
{
    uint8_t barIdx = (uint8_t)args->scalarInput[0];
    uint8_t memIdx = 0; uint64_t barSz = 0;
    if (getBarMemoryIndex(iv->pciDevice, barIdx, &memIdx, &barSz) != kIOReturnSuccess)
        return kIOReturnNotFound;
    args->scalarOutput[0] = barSz;
    return kIOReturnSuccess;
}

static kern_return_t handleAllocDMA(ROCmGPUUserClient_IVars* iv,
                                    IOUserClientMethodArguments* args)
{
    uint64_t size = args->scalarInput[0];
    if (size == 0 || size > (1ULL << 32)) return kIOReturnBadArgument;

    uint32_t bufID = UINT32_MAX;
    for (uint32_t i = 0; i < MAX_DMA_BUFFERS; i++) {
        uint32_t idx = (iv->nextDMAID + i) % MAX_DMA_BUFFERS;
        if (!iv->dmaBuffers[idx].inUse) {
            bufID = idx;
            iv->nextDMAID = (idx + 1) % MAX_DMA_BUFFERS;
            break;
        }
    }
    if (bufID == UINT32_MAX) return kIOReturnNoResources;

    auto& buf = iv->dmaBuffers[bufID];
    buf.inUse = true;
    kern_return_t ret;

    ret = IOBufferMemoryDescriptor::Create(
        kIOMemoryDirectionOutIn, size, 0, &buf.descriptor);
    if (ret != kIOReturnSuccess) {
        buf.inUse = false;
        return ret;
    }

    IODMACommandSpecification spec = {};
    spec.maxAddressBits = 64;
    ret = IODMACommand::Create(iv->pciDevice, 0, &spec, &buf.dmaCommand);
    if (ret != kIOReturnSuccess) {
        OSSafeReleaseNULL(buf.descriptor);
        buf.inUse = false;
        return ret;
    }

    uint64_t dmaFlags = 0;
    uint32_t segCount = 32;
    IOAddressSegment segments[32] = {};
    ret = buf.dmaCommand->PrepareForDMA(0, buf.descriptor, 0, size,
                                         &dmaFlags, &segCount, segments);
    if (ret != kIOReturnSuccess) {
        OSSafeReleaseNULL(buf.dmaCommand);
        OSSafeReleaseNULL(buf.descriptor);
        buf.inUse = false;
        return ret;
    }

    buf.physAddr = segments[0].address;
    buf.size = size;

    ROCmGPUDMAInfo info = {};
    info.bufferID = bufID;
    info.size = size;
    info.segmentCount = (segCount > 64) ? 64 : segCount;
    for (uint32_t i = 0; i < info.segmentCount && i < 32; i++) {
        info.segments[i].address = segments[i].address;
        info.segments[i].length = segments[i].length;
    }

    args->structureOutput = OSData::withBytes(&info, sizeof(info));
    return args->structureOutput ? kIOReturnSuccess : kIOReturnNoMemory;
}

static kern_return_t handleFreeDMA(ROCmGPUUserClient_IVars* iv,
                                   IOUserClientMethodArguments* args)
{
    uint32_t bufID = (uint32_t)args->scalarInput[0];
    if (bufID >= MAX_DMA_BUFFERS || !iv->dmaBuffers[bufID].inUse)
        return kIOReturnNotFound;

    auto& buf = iv->dmaBuffers[bufID];
    if (buf.dmaCommand) { buf.dmaCommand->CompleteDMA(0); OSSafeReleaseNULL(buf.dmaCommand); }
    OSSafeReleaseNULL(buf.descriptor);
    buf = {};
    return kIOReturnSuccess;
}

static kern_return_t handleMapDMA(ROCmGPUUserClient_IVars* iv,
                                  IOUserClientMethodArguments* args)
{
    uint32_t bufID = (uint32_t)args->scalarInput[0];
    if (bufID >= MAX_DMA_BUFFERS || !iv->dmaBuffers[bufID].inUse)
        return kIOReturnNotFound;
    args->scalarOutput[0] = iv->dmaBuffers[bufID].size;
    return kIOReturnSuccess;
}

static kern_return_t handleEnableMSI(ROCmGPUUserClient_IVars* iv,
                                     IOUserClientMethodArguments* args)
{
    if (iv->intSource) return kIOReturnStillOpen;

    IODispatchQueue* queue = nullptr;
    kern_return_t ret = IODispatchQueue::Create("ROCmGPU-Int", 0, 0, &queue);
    if (ret != kIOReturnSuccess) return ret;

    ret = IOInterruptDispatchSource::Create(
        iv->pciDevice, (uint32_t)args->scalarInput[0], queue, &iv->intSource);
    OSSafeReleaseNULL(queue);
    if (ret != kIOReturnSuccess) return ret;

    iv->intSource->SetEnable(true);
    iv->intFired = false;
    return kIOReturnSuccess;
}

static kern_return_t handleWaitInterrupt(ROCmGPUUserClient_IVars* iv,
                                         IOUserClientMethodArguments* args)
{
    args->scalarOutput[0] = iv->intFired ? kROCmGPU_IntStatus_OK : kROCmGPU_IntStatus_Timeout;
    iv->intFired = false;
    return kIOReturnSuccess;
}

/* ======================================================================
 * Lifecycle
 * ====================================================================== */

bool ROCmGPUUserClient::init()
{
    if (!super::init()) return false;
    ivars = IONewZero(ROCmGPUUserClient_IVars, 1);
    return ivars != nullptr;
}

kern_return_t ROCmGPUUserClient::Start_Impl(IOService* provider)
{
    kern_return_t ret = Start(provider, SUPERDISPATCH);
    if (ret != kIOReturnSuccess) return ret;

    ivars->driver = OSDynamicCast(ROCmGPUDriver, provider);
    if (!ivars->driver) return kIOReturnError;

    /* Get PCI device from our provider's provider (the IOPCIDevice) */
    ivars->pciDevice = OSDynamicCast(IOPCIDevice, ivars->driver->GetProvider());
    if (!ivars->pciDevice) return kIOReturnNoDevice;

    os_log(OS_LOG_DEFAULT, LOG_PREFIX "UserClient started");
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::Stop_Impl(IOService* provider)
{
    for (uint32_t i = 0; i < MAX_DMA_BUFFERS; i++) {
        auto& buf = ivars->dmaBuffers[i];
        if (buf.inUse) {
            if (buf.dmaCommand) { buf.dmaCommand->CompleteDMA(0); OSSafeReleaseNULL(buf.dmaCommand); }
            OSSafeReleaseNULL(buf.descriptor);
            buf.inUse = false;
        }
    }
    OSSafeReleaseNULL(ivars->intSource);
    return Stop(provider, SUPERDISPATCH);
}

void ROCmGPUUserClient::free()
{
    IOSafeDeleteNULL(ivars, ROCmGPUUserClient_IVars, 1);
    super::free();
}

/* ======================================================================
 * ExternalMethod — virtual override (dispatch to static handlers)
 * ====================================================================== */

kern_return_t ROCmGPUUserClient::ExternalMethod(
    uint64_t selector,
    IOUserClientMethodArguments* args,
    const IOUserClientMethodDispatch* dispatch,
    OSObject* target,
    void* reference)
{
    if (selector >= kROCmGPU_SelectorCount) return kIOReturnBadArgument;

    switch (selector) {
    case kROCmGPU_GetInfo:       return handleGetInfo(ivars, args);
    case kROCmGPU_Reset:         return handleReset(ivars, args);
    case kROCmGPU_CfgRead:       return handleCfgRead(ivars, args);
    case kROCmGPU_CfgWrite:      return handleCfgWrite(ivars, args);
    case kROCmGPU_MMIORead32:    return handleMMIORead32(ivars, args);
    case kROCmGPU_MMIOWrite32:   return handleMMIOWrite32(ivars, args);
    case kROCmGPU_MapBAR:        return handleMapBAR(ivars, args);
    case kROCmGPU_UnmapBAR:      return kIOReturnSuccess;
    case kROCmGPU_AllocDMA:      return handleAllocDMA(ivars, args);
    case kROCmGPU_FreeDMA:       return handleFreeDMA(ivars, args);
    case kROCmGPU_MapDMA:        return handleMapDMA(ivars, args);
    case kROCmGPU_EnableMSI:     return handleEnableMSI(ivars, args);
    case kROCmGPU_WaitInterrupt: return handleWaitInterrupt(ivars, args);
    default: return kIOReturnBadArgument;
    }
}

/* ======================================================================
 * CopyClientMemoryForType_Impl
 * ====================================================================== */

kern_return_t ROCmGPUUserClient::CopyClientMemoryForType_Impl(
    uint64_t type,
    uint64_t* options,
    IOMemoryDescriptor** memory)
{
    os_log(OS_LOG_DEFAULT, LOG_PREFIX "CopyClientMemoryForType(type=0x%llx) called", type);

    if (type <= kROCmGPU_MemType_BAR5) {
        uint8_t memIdx = 0; uint64_t barSz = 0;
        kern_return_t ret = getBarMemoryIndex(ivars->pciDevice, (uint8_t)type, &memIdx, &barSz);
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "  GetBARInfo(%u) ret=0x%x memIdx=%u size=%llu",
               (unsigned)type, ret, memIdx, barSz);
        if (ret != kIOReturnSuccess)
            return kIOReturnNotFound;

        if (options) {
            *options = (*options & ~0xF00ULL) | kIOMemoryMapCacheModeInhibit;
        }

        // _CopyDeviceMemoryWithIndex's `forClient` parameter must be an
        // IOService attached to the PCI device. Our UserClient is attached
        // to the Driver (which is itself attached to the PCI device), so
        // `this` here would fail with kIOReturnNotAttached (0xE00002CD).
        // Pass the Driver instead — same approach tinygrad's TinyGPU uses.
        IOMemoryDescriptor* desc = nullptr;
        ret = ivars->pciDevice->_CopyDeviceMemoryWithIndex(
            memIdx, &desc, ivars->driver);
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "  _CopyDeviceMemoryWithIndex(%u) ret=0x%x desc=%p",
               memIdx, ret, desc);
        if (ret != kIOReturnSuccess || !desc) return kIOReturnError;

        *memory = desc;
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "  OK returning desc=%p for BAR %u", desc, (unsigned)type);
        return kIOReturnSuccess;

    } else if (type >= kROCmGPU_MemType_DMABase) {
        uint32_t bufID = (uint32_t)(type - kROCmGPU_MemType_DMABase);
        os_log(OS_LOG_DEFAULT, LOG_PREFIX "  DMA buffer path, bufID=%u", bufID);
        if (bufID >= MAX_DMA_BUFFERS || !ivars->dmaBuffers[bufID].inUse)
            return kIOReturnNotFound;

        ivars->dmaBuffers[bufID].descriptor->retain();
        *memory = ivars->dmaBuffers[bufID].descriptor;
        return kIOReturnSuccess;
    }

    os_log(OS_LOG_DEFAULT, LOG_PREFIX "  type=0x%llx unrecognized, returning kIOReturnBadArgument", type);
    return kIOReturnBadArgument;
}
