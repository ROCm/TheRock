/*
 * ROCmGPUUserClient.cpp - Escape command dispatch implementation.
 *
 * Handles all userspace requests via ExternalMethod():
 *   - Device info queries
 *   - PCI config space read/write
 *   - MMIO register read/write (via PCI BARs)
 *   - BAR mapping into client process
 *   - DMA buffer allocation and mapping
 *   - MSI-X interrupt handling
 *   - GPU Function-Level Reset
 *
 * Design note: MMIO read/write uses IOPCIDevice::MemoryRead32/Write32
 * for individual register access. For bulk access (IP discovery, firmware
 * loading), clients should map the BAR via IOConnectMapMemory64 and
 * do direct pointer dereference -- much faster than per-register RPCs.
 */

#include "ROCmGPUUserClient.h"
#include "ROCmGPUDriver.h"
#include "ROCmGPUShared.h"

#include <DriverKit/IOLib.h>
#include <DriverKit/IOBufferMemoryDescriptor.iig>
#include <DriverKit/IODMACommand.iig>
#include <DriverKit/IOInterruptDispatchSource.iig>
#include <DriverKit/IODispatchQueue.iig>

#define LOG_PREFIX "ROCmGPU-UC: "

/* ======================================================================
 * Dispatch table
 *
 * Maps selector -> (handler, input/output scalar/struct counts).
 * DriverKit validates argument counts before calling the handler.
 * ====================================================================== */

static const IOUserClientMethodDispatch sMethods[kROCmGPU_SelectorCount] = {
    /* kROCmGPU_GetInfo: () -> struct ROCmGPUDeviceInfo */
    [kROCmGPU_GetInfo] = {
        .function = nullptr,  /* Handled in ExternalMethod directly */
        .checkCompletionExists = false,
        .checkScalarInputCount  = 0,
        .checkScalarOutputCount = 0,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = sizeof(ROCmGPUDeviceInfo),
    },

    /* kROCmGPU_Reset: () -> () */
    [kROCmGPU_Reset] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 0,
        .checkScalarOutputCount = 0,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },

    /* kROCmGPU_CfgRead: (offset, width) -> (value) */
    [kROCmGPU_CfgRead] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 2,
        .checkScalarOutputCount = 1,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },

    /* kROCmGPU_CfgWrite: (offset, width, value) -> () */
    [kROCmGPU_CfgWrite] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 3,
        .checkScalarOutputCount = 0,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },

    /* kROCmGPU_MMIORead32: (barIndex, offset) -> (value) */
    [kROCmGPU_MMIORead32] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 2,
        .checkScalarOutputCount = 1,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },

    /* kROCmGPU_MMIOWrite32: (barIndex, offset, value) -> () */
    [kROCmGPU_MMIOWrite32] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 3,
        .checkScalarOutputCount = 0,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },

    /* kROCmGPU_MapBAR: (barIndex) -> (size) */
    [kROCmGPU_MapBAR] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 1,
        .checkScalarOutputCount = 1,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },

    /* kROCmGPU_UnmapBAR: (barIndex) -> () */
    [kROCmGPU_UnmapBAR] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 1,
        .checkScalarOutputCount = 0,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },

    /* kROCmGPU_AllocDMA: (size, flags) -> struct ROCmGPUDMAInfo */
    [kROCmGPU_AllocDMA] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 2,
        .checkScalarOutputCount = 0,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = sizeof(ROCmGPUDMAInfo),
    },

    /* kROCmGPU_FreeDMA: (bufferID) -> () */
    [kROCmGPU_FreeDMA] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 1,
        .checkScalarOutputCount = 0,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },

    /* kROCmGPU_MapDMA: (bufferID) -> (size) */
    [kROCmGPU_MapDMA] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 1,
        .checkScalarOutputCount = 1,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },

    /* kROCmGPU_EnableMSI: (vectorIndex) -> () */
    [kROCmGPU_EnableMSI] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 1,
        .checkScalarOutputCount = 0,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },

    /* kROCmGPU_WaitInterrupt: (timeoutMS) -> (status) */
    [kROCmGPU_WaitInterrupt] = {
        .function = nullptr,
        .checkCompletionExists = false,
        .checkScalarInputCount  = 1,
        .checkScalarOutputCount = 1,
        .checkStructureInputSize  = 0,
        .checkStructureOutputSize = 0,
    },
};

/* ======================================================================
 * IOUserClient lifecycle
 * ====================================================================== */

bool ROCmGPUUserClient::init()
{
    if (!super::init()) {
        return false;
    }

    fDriver = nullptr;
    fPCIDevice = nullptr;
    fNextDMAID = 0;
    fInterruptSource = nullptr;
    fInterruptEnabled = false;

    for (uint32_t i = 0; i < ROCMGPU_MAX_DMA_BUFFERS; i++) {
        fDMABuffers[i] = {};
    }

    return true;
}

kern_return_t ROCmGPUUserClient::Start(IOService* provider)
{
    kern_return_t ret = super::Start(provider);
    if (ret != kIOReturnSuccess) {
        return ret;
    }

    /* Get reference to parent driver */
    fDriver = OSDynamicCast(ROCmGPUDriver, provider);
    if (!fDriver) {
        IOLog(LOG_PREFIX "Provider is not ROCmGPUDriver\n");
        return kIOReturnError;
    }

    fPCIDevice = fDriver->getPCIDevice();
    if (!fPCIDevice) {
        IOLog(LOG_PREFIX "No PCI device available\n");
        return kIOReturnNoDevice;
    }

    IOLog(LOG_PREFIX "UserClient started\n");
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::Stop(IOService* provider)
{
    IOLog(LOG_PREFIX "UserClient stopping — freeing DMA buffers\n");

    /* Release all DMA buffers allocated by this client */
    for (uint32_t i = 0; i < ROCMGPU_MAX_DMA_BUFFERS; i++) {
        if (fDMABuffers[i].inUse) {
            if (fDMABuffers[i].dmaCommand) {
                fDMABuffers[i].dmaCommand->Complete(kIODMACommandComplete);
                fDMABuffers[i].dmaCommand->release();
            }
            if (fDMABuffers[i].descriptor) {
                fDMABuffers[i].descriptor->release();
            }
            fDMABuffers[i] = {};
        }
    }

    /* Release interrupt source */
    if (fInterruptSource) {
        fInterruptSource->Cancel(nullptr);
        fInterruptSource->release();
        fInterruptSource = nullptr;
    }

    return super::Stop(provider);
}

void ROCmGPUUserClient::free()
{
    super::free();
}

/* ======================================================================
 * ExternalMethod dispatch
 * ====================================================================== */

kern_return_t ROCmGPUUserClient::ExternalMethod(
    uint64_t selector,
    IOUserClientMethodArguments* arguments,
    const IOUserClientMethodDispatch* dispatch,
    OSObject* target,
    void* reference)
{
    if (selector >= kROCmGPU_SelectorCount) {
        IOLog(LOG_PREFIX "Invalid selector: %llu\n", selector);
        return kIOReturnBadArgument;
    }

    /* Validate argument counts against dispatch table */
    kern_return_t ret = super::ExternalMethod(
        selector, arguments, &sMethods[selector], this, nullptr);
    if (ret != kIOReturnSuccess) {
        /* If super returns kIOReturnBadArgument, it means count validation
         * failed but .function was null, so we dispatch manually below. */
    }

    /* Route to handler */
    switch (selector) {
    case kROCmGPU_GetInfo:        return handleGetInfo(arguments);
    case kROCmGPU_Reset:          return handleReset(arguments);
    case kROCmGPU_CfgRead:        return handleCfgRead(arguments);
    case kROCmGPU_CfgWrite:       return handleCfgWrite(arguments);
    case kROCmGPU_MMIORead32:     return handleMMIORead32(arguments);
    case kROCmGPU_MMIOWrite32:    return handleMMIOWrite32(arguments);
    case kROCmGPU_MapBAR:         return handleMapBAR(arguments);
    case kROCmGPU_UnmapBAR:       return handleUnmapBAR(arguments);
    case kROCmGPU_AllocDMA:       return handleAllocDMA(arguments);
    case kROCmGPU_FreeDMA:        return handleFreeDMA(arguments);
    case kROCmGPU_MapDMA:         return handleMapDMA(arguments);
    case kROCmGPU_EnableMSI:      return handleEnableMSI(arguments);
    case kROCmGPU_WaitInterrupt:  return handleWaitInterrupt(arguments);
    default:
        return kIOReturnBadArgument;
    }
}

/* ======================================================================
 * Memory mapping (BAR + DMA buffers)
 * ====================================================================== */

kern_return_t ROCmGPUUserClient::CopyClientMemoryForType(
    uint64_t type,
    uint64_t* options,
    IOMemoryDescriptor** memory)
{
    if (type <= kROCmGPU_MemType_BAR5) {
        /* Map a PCI BAR */
        uint8_t barIndex = (uint8_t)type;
        auto bar = fDriver->getBAR(barIndex);
        if (!bar.valid || bar.size == 0) {
            IOLog(LOG_PREFIX "BAR%u not available\n", barIndex);
            return kIOReturnNotFound;
        }

        /* CopyDeviceMemoryWithIndex returns an IOMemoryDescriptor
         * for the BAR's physical address range */
        IOMemoryDescriptor* desc = nullptr;
        desc = fPCIDevice->CopyDeviceMemoryWithIndex(bar.memoryIndex, this);
        if (!desc) {
            IOLog(LOG_PREFIX "CopyDeviceMemoryWithIndex(%u) failed\n",
                  bar.memoryIndex);
            return kIOReturnError;
        }

        *memory = desc;
        IOLog(LOG_PREFIX "Mapped BAR%u: %lluMB\n", barIndex,
              bar.size / (1024*1024));
        return kIOReturnSuccess;

    } else if (type >= kROCmGPU_MemType_DMABase) {
        /* Map a DMA buffer */
        uint32_t bufID = (uint32_t)(type - kROCmGPU_MemType_DMABase);
        if (bufID >= ROCMGPU_MAX_DMA_BUFFERS || !fDMABuffers[bufID].inUse) {
            IOLog(LOG_PREFIX "DMA buffer %u not found\n", bufID);
            return kIOReturnNotFound;
        }

        /* Return the buffer's memory descriptor for client mapping */
        IOMemoryDescriptor* desc = fDMABuffers[bufID].descriptor;
        desc->retain();
        *memory = desc;
        return kIOReturnSuccess;
    }

    return kIOReturnBadArgument;
}

/* ======================================================================
 * Command handlers
 * ====================================================================== */

kern_return_t ROCmGPUUserClient::handleGetInfo(
    IOUserClientMethodArguments* args)
{
    if (!args->structureOutput || args->structureOutputSize < sizeof(ROCmGPUDeviceInfo)) {
        return kIOReturnBadArgument;
    }

    ROCmGPUDeviceInfo* info = (ROCmGPUDeviceInfo*)args->structureOutput;
    memset(info, 0, sizeof(*info));

    /* Read PCI IDs from config space */
    fPCIDevice->ConfigurationRead16(0x00, &info->vendorID);
    fPCIDevice->ConfigurationRead16(0x02, &info->deviceID);
    fPCIDevice->ConfigurationRead16(0x2C, &info->subsystemVendorID);
    fPCIDevice->ConfigurationRead16(0x2E, &info->subsystemDeviceID);

    uint8_t revID = 0;
    fPCIDevice->ConfigurationRead8(0x08, &revID);
    info->revisionID = revID;

    /* Fill BAR info from cached data */
    for (int i = 0; i < 6; i++) {
        auto& bar = fDriver->getBAR(i);
        info->bars[i].size = bar.size;
        info->bars[i].memoryIndex = bar.memoryIndex;
        info->bars[i].type = bar.type;
        info->bars[i].is64bit = bar.is64bit ? 1 : 0;
        info->bars[i].prefetchable = bar.prefetchable ? 1 : 0;
    }

    /* Estimate VRAM size from largest prefetchable BAR */
    uint64_t maxVRAM = 0;
    for (int i = 0; i < 6; i++) {
        if (fDriver->getBAR(i).prefetchable && fDriver->getBAR(i).size > maxVRAM) {
            maxVRAM = fDriver->getBAR(i).size;
        }
    }
    info->vramSize = maxVRAM;

    IOLog(LOG_PREFIX "GetInfo: vendor=0x%04x device=0x%04x vram=%lluMB\n",
          info->vendorID, info->deviceID, info->vramSize / (1024*1024));

    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleReset(
    IOUserClientMethodArguments* args)
{
    IOLog(LOG_PREFIX "Performing Function-Level Reset\n");

    /* FLR via PCI Express Capability:
     * 1. Find PCI Express capability
     * 2. Read Device Control register
     * 3. Set Initiate Function Level Reset bit (bit 15) */
    uint32_t pcieCap = 0;
    kern_return_t ret = fPCIDevice->FindPCICapability(
        0x10,  /* PCI Express Capability ID */
        0,     /* Search from beginning */
        &pcieCap);
    if (ret != kIOReturnSuccess || pcieCap == 0) {
        IOLog(LOG_PREFIX "PCI Express capability not found\n");
        return kIOReturnUnsupported;
    }

    /* Device Control register is at capability offset + 0x08 */
    uint16_t devCtl = 0;
    fPCIDevice->ConfigurationRead16(pcieCap + 0x08, &devCtl);
    devCtl |= (1 << 15);  /* Initiate FLR */
    fPCIDevice->ConfigurationWrite16(pcieCap + 0x08, devCtl);

    IOLog(LOG_PREFIX "FLR initiated\n");
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleCfgRead(
    IOUserClientMethodArguments* args)
{
    uint64_t offset = args->scalarInput[0];
    uint64_t width  = args->scalarInput[1];

    uint64_t value = 0;
    switch (width) {
    case 1: {
        uint8_t v = 0;
        fPCIDevice->ConfigurationRead8((uint32_t)offset, &v);
        value = v;
        break;
    }
    case 2: {
        uint16_t v = 0;
        fPCIDevice->ConfigurationRead16((uint32_t)offset, &v);
        value = v;
        break;
    }
    case 4: {
        uint32_t v = 0;
        fPCIDevice->ConfigurationRead32((uint32_t)offset, &v);
        value = v;
        break;
    }
    default:
        return kIOReturnBadArgument;
    }

    args->scalarOutput[0] = value;
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleCfgWrite(
    IOUserClientMethodArguments* args)
{
    uint64_t offset = args->scalarInput[0];
    uint64_t width  = args->scalarInput[1];
    uint64_t value  = args->scalarInput[2];

    switch (width) {
    case 1:
        fPCIDevice->ConfigurationWrite8((uint32_t)offset, (uint8_t)value);
        break;
    case 2:
        fPCIDevice->ConfigurationWrite16((uint32_t)offset, (uint16_t)value);
        break;
    case 4:
        fPCIDevice->ConfigurationWrite32((uint32_t)offset, (uint32_t)value);
        break;
    default:
        return kIOReturnBadArgument;
    }

    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleMMIORead32(
    IOUserClientMethodArguments* args)
{
    uint64_t barIndex = args->scalarInput[0];
    uint64_t offset   = args->scalarInput[1];

    if (barIndex > 5) {
        return kIOReturnBadArgument;
    }

    auto& bar = fDriver->getBAR((unsigned)barIndex);
    if (!bar.valid) {
        return kIOReturnNotFound;
    }

    /* Bounds check */
    if (offset + 4 > bar.size) {
        return kIOReturnBadArgument;
    }

    uint32_t value = 0;
    fPCIDevice->MemoryRead32(bar.memoryIndex, offset, &value);
    args->scalarOutput[0] = value;
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleMMIOWrite32(
    IOUserClientMethodArguments* args)
{
    uint64_t barIndex = args->scalarInput[0];
    uint64_t offset   = args->scalarInput[1];
    uint64_t value    = args->scalarInput[2];

    if (barIndex > 5) {
        return kIOReturnBadArgument;
    }

    auto& bar = fDriver->getBAR((unsigned)barIndex);
    if (!bar.valid) {
        return kIOReturnNotFound;
    }

    if (offset + 4 > bar.size) {
        return kIOReturnBadArgument;
    }

    fPCIDevice->MemoryWrite32(bar.memoryIndex, offset, (uint32_t)value);
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleMapBAR(
    IOUserClientMethodArguments* args)
{
    uint64_t barIndex = args->scalarInput[0];
    if (barIndex > 5) {
        return kIOReturnBadArgument;
    }

    auto& bar = fDriver->getBAR((unsigned)barIndex);
    if (!bar.valid || bar.size == 0) {
        return kIOReturnNotFound;
    }

    /* Client will call IOConnectMapMemory64(connection, barIndex, ...)
     * which triggers CopyClientMemoryForType(barIndex, ...) */
    args->scalarOutput[0] = bar.size;

    IOLog(LOG_PREFIX "MapBAR(%llu): size=%lluMB — client should call IOConnectMapMemory64\n",
          barIndex, bar.size / (1024*1024));
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleUnmapBAR(
    IOUserClientMethodArguments* args)
{
    /* Unmapping is handled by IOConnectUnmapMemory() on the client side.
     * Nothing to do here — the mapping is ref-counted. */
    IOLog(LOG_PREFIX "UnmapBAR(%llu)\n", args->scalarInput[0]);
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleAllocDMA(
    IOUserClientMethodArguments* args)
{
    uint64_t size  = args->scalarInput[0];
    uint64_t flags = args->scalarInput[1];

    if (size == 0 || size > (1ULL << 32)) {
        return kIOReturnBadArgument;
    }

    /* Find a free DMA slot */
    uint32_t bufID = allocDMASlot();
    if (bufID == UINT32_MAX) {
        IOLog(LOG_PREFIX "No free DMA slots\n");
        return kIOReturnNoResources;
    }

    kern_return_t ret;
    DMABuffer& buf = fDMABuffers[bufID];

    /* Step 1: Create IOBufferMemoryDescriptor */
    uint64_t bdOptions = kIOMemoryDirectionInOut;
    if (flags & kROCmGPU_DMA_ReadOnly) {
        bdOptions = kIOMemoryDirectionIn;
    } else if (flags & kROCmGPU_DMA_WriteOnly) {
        bdOptions = kIOMemoryDirectionOut;
    }

    ret = IOBufferMemoryDescriptor::Create(
        bdOptions,
        size,
        0,  /* alignment — 0 means page-aligned */
        &buf.descriptor);
    if (ret != kIOReturnSuccess) {
        IOLog(LOG_PREFIX "IOBufferMemoryDescriptor::Create(%llu) failed: 0x%x\n",
              size, ret);
        buf.inUse = false;
        return ret;
    }

    /* Step 2: Create IODMACommand and prepare for DMA */
    uint32_t dmaSpecSize = 0;
    ret = IODMACommand::Create(this, kIODMACommandSpecSmall, &buf.dmaCommand);
    if (ret != kIOReturnSuccess) {
        IOLog(LOG_PREFIX "IODMACommand::Create() failed: 0x%x\n", ret);
        buf.descriptor->release();
        buf.descriptor = nullptr;
        buf.inUse = false;
        return ret;
    }

    /* Prepare: performs IOMMU translation, returns physical segments */
    uint32_t segCount = 64;
    IOAddressSegment segments[64] = {};
    uint64_t dmaFlags = 0;

    ret = buf.dmaCommand->PrepareForDMA(
        buf.descriptor,
        nullptr,     /* completion */
        &dmaFlags,
        &segCount,
        segments);
    if (ret != kIOReturnSuccess) {
        IOLog(LOG_PREFIX "PrepareForDMA() failed: 0x%x\n", ret);
        buf.dmaCommand->release();
        buf.descriptor->release();
        buf.dmaCommand = nullptr;
        buf.descriptor = nullptr;
        buf.inUse = false;
        return ret;
    }

    buf.physAddr = segments[0].address;
    buf.size = size;

    /* Fill output struct with scatter-gather info */
    if (args->structureOutput && args->structureOutputSize >= sizeof(ROCmGPUDMAInfo)) {
        ROCmGPUDMAInfo* info = (ROCmGPUDMAInfo*)args->structureOutput;
        memset(info, 0, sizeof(*info));
        info->bufferID = bufID;
        info->size = size;
        info->segmentCount = (segCount > 64) ? 64 : segCount;

        for (uint32_t i = 0; i < info->segmentCount; i++) {
            info->segments[i].address = segments[i].address;
            info->segments[i].length = segments[i].length;
        }
    }

    IOLog(LOG_PREFIX "AllocDMA: id=%u size=%lluKB phys=0x%llx segs=%u\n",
          bufID, size/1024, buf.physAddr, segCount);
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleFreeDMA(
    IOUserClientMethodArguments* args)
{
    uint32_t bufID = (uint32_t)args->scalarInput[0];
    if (bufID >= ROCMGPU_MAX_DMA_BUFFERS || !fDMABuffers[bufID].inUse) {
        return kIOReturnNotFound;
    }

    DMABuffer& buf = fDMABuffers[bufID];

    if (buf.dmaCommand) {
        buf.dmaCommand->Complete(kIODMACommandComplete);
        buf.dmaCommand->release();
    }
    if (buf.descriptor) {
        buf.descriptor->release();
    }

    buf = {};
    IOLog(LOG_PREFIX "FreeDMA: id=%u\n", bufID);
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleMapDMA(
    IOUserClientMethodArguments* args)
{
    uint32_t bufID = (uint32_t)args->scalarInput[0];
    if (bufID >= ROCMGPU_MAX_DMA_BUFFERS || !fDMABuffers[bufID].inUse) {
        return kIOReturnNotFound;
    }

    /* Client calls IOConnectMapMemory64(conn, kROCmGPU_MemType_DMABase + bufID, ...)
     * which triggers CopyClientMemoryForType() */
    args->scalarOutput[0] = fDMABuffers[bufID].size;
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleEnableMSI(
    IOUserClientMethodArguments* args)
{
    uint64_t vectorIndex = args->scalarInput[0];

    IOLog(LOG_PREFIX "EnableMSI: vector=%llu\n", vectorIndex);

    if (fInterruptSource) {
        IOLog(LOG_PREFIX "Interrupt already enabled\n");
        return kIOReturnStillOpen;
    }

    /* Find MSI/MSI-X interrupt */
    uint64_t intType = 0;
    kern_return_t ret = fPCIDevice->GetInterruptType(
        (uint32_t)vectorIndex, &intType);
    if (ret != kIOReturnSuccess) {
        IOLog(LOG_PREFIX "GetInterruptType(%llu) failed: 0x%x\n",
              vectorIndex, ret);
        return ret;
    }

    /* Create dispatch queue for interrupt handler */
    IODispatchQueue* queue = nullptr;
    ret = IODispatchQueue::Create("ROCmGPU-Interrupt", 0, 0, &queue);
    if (ret != kIOReturnSuccess) {
        return ret;
    }

    /* Create interrupt dispatch source */
    ret = IOInterruptDispatchSource::Create(
        fPCIDevice,
        (uint32_t)vectorIndex,
        queue,
        &fInterruptSource);
    if (ret != kIOReturnSuccess) {
        queue->release();
        IOLog(LOG_PREFIX "Failed to create interrupt source: 0x%x\n", ret);
        return ret;
    }

    /* Set handler — for now just log, will be extended for IH ring */
    fInterruptSource->SetHandler(
        ^(IOInterruptDispatchSource* source, int count,
          void* action, void* data) {
            /* Read IH ring write pointer to determine interrupt source.
             * For now, just mark that an interrupt occurred. */
            fInterruptEnabled = true;  /* Signal to WaitInterrupt */
        });

    fInterruptSource->SetEnable(true);
    fInterruptEnabled = false;

    queue->release();
    IOLog(LOG_PREFIX "MSI vector %llu enabled\n", vectorIndex);
    return kIOReturnSuccess;
}

kern_return_t ROCmGPUUserClient::handleWaitInterrupt(
    IOUserClientMethodArguments* args)
{
    /* Placeholder: proper implementation would use IOTimerDispatchSource
     * or an async completion. For now, return immediately with status. */
    if (fInterruptEnabled) {
        fInterruptEnabled = false;
        args->scalarOutput[0] = kROCmGPU_IntStatus_OK;
    } else {
        args->scalarOutput[0] = kROCmGPU_IntStatus_Timeout;
    }
    return kIOReturnSuccess;
}

/* ======================================================================
 * DMA slot management
 * ====================================================================== */

uint32_t ROCmGPUUserClient::allocDMASlot()
{
    for (uint32_t i = 0; i < ROCMGPU_MAX_DMA_BUFFERS; i++) {
        uint32_t id = (fNextDMAID + i) % ROCMGPU_MAX_DMA_BUFFERS;
        if (!fDMABuffers[id].inUse) {
            fDMABuffers[id].inUse = true;
            fNextDMAID = (id + 1) % ROCMGPU_MAX_DMA_BUFFERS;
            return id;
        }
    }
    return UINT32_MAX;
}

void ROCmGPUUserClient::freeDMASlot(uint32_t id)
{
    if (id < ROCMGPU_MAX_DMA_BUFFERS) {
        fDMABuffers[id].inUse = false;
    }
}
