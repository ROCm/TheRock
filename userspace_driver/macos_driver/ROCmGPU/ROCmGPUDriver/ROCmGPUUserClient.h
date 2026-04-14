/*
 * ROCmGPUUserClient.h - Per-client escape command dispatch.
 *
 * Each userspace connection gets its own ROCmGPUUserClient instance.
 * It dispatches external method calls (IOConnectCallScalarMethod /
 * IOConnectCallStructMethod) to handler functions that perform
 * PCI operations on behalf of the client.
 *
 * Memory mapping (BAR and DMA buffers) uses CopyClientMemoryForType()
 * so clients can call IOConnectMapMemory64() to get direct access.
 */

#ifndef ROCMGPU_USERCLIENT_H
#define ROCMGPU_USERCLIENT_H

#include <DriverKit/IOUserClient.iig>
#include <DriverKit/IOBufferMemoryDescriptor.iig>
#include <DriverKit/IODMACommand.iig>
#include <PCIDriverKit/IOPCIDevice.iig>

#include "ROCmGPUShared.h"

class ROCmGPUDriver;

/* Maximum number of DMA buffers a single client can allocate */
#define ROCMGPU_MAX_DMA_BUFFERS 256

class ROCmGPUUserClient : public IOUserClient
{
public:
    /* IOUserClient lifecycle */
    virtual bool init() override;
    virtual kern_return_t Start(IOService* provider) override;
    virtual kern_return_t Stop(IOService* provider) override;
    virtual void free() override;

    /* Escape command dispatch */
    virtual kern_return_t ExternalMethod(
        uint64_t selector,
        IOUserClientMethodArguments* arguments,
        const IOUserClientMethodDispatch* dispatch,
        OSObject* target,
        void* reference) override;

    /* Memory mapping for BAR and DMA regions */
    virtual kern_return_t CopyClientMemoryForType(
        uint64_t type,
        uint64_t* options,
        IOMemoryDescriptor** memory) override;

private:
    /* Parent driver reference */
    ROCmGPUDriver* fDriver;
    IOPCIDevice*   fPCIDevice;

    /* DMA buffer tracking */
    struct DMABuffer {
        IOBufferMemoryDescriptor* descriptor;
        IODMACommand*             dmaCommand;
        uint64_t                  physAddr;    /* First segment IOVA */
        uint64_t                  size;
        bool                      inUse;
    };

    DMABuffer fDMABuffers[ROCMGPU_MAX_DMA_BUFFERS];
    uint32_t  fNextDMAID;

    /* MSI-X interrupt state */
    IOInterruptDispatchSource* fInterruptSource;
    bool fInterruptEnabled;

    /* ---- Escape command handlers ---- */

    kern_return_t handleGetInfo(
        IOUserClientMethodArguments* args);

    kern_return_t handleReset(
        IOUserClientMethodArguments* args);

    kern_return_t handleCfgRead(
        IOUserClientMethodArguments* args);

    kern_return_t handleCfgWrite(
        IOUserClientMethodArguments* args);

    kern_return_t handleMMIORead32(
        IOUserClientMethodArguments* args);

    kern_return_t handleMMIOWrite32(
        IOUserClientMethodArguments* args);

    kern_return_t handleMapBAR(
        IOUserClientMethodArguments* args);

    kern_return_t handleUnmapBAR(
        IOUserClientMethodArguments* args);

    kern_return_t handleAllocDMA(
        IOUserClientMethodArguments* args);

    kern_return_t handleFreeDMA(
        IOUserClientMethodArguments* args);

    kern_return_t handleMapDMA(
        IOUserClientMethodArguments* args);

    kern_return_t handleEnableMSI(
        IOUserClientMethodArguments* args);

    kern_return_t handleWaitInterrupt(
        IOUserClientMethodArguments* args);

    /* DMA helpers */
    uint32_t allocDMASlot();
    void     freeDMASlot(uint32_t id);
};

#endif /* ROCMGPU_USERCLIENT_H */
