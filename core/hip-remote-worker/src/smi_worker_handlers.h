/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file smi_worker_handlers.h
 * @brief AMD SMI request handlers for the HIP remote worker
 *
 * These handlers implement the server-side logic for remote AMD SMI
 * operations, allowing clients to query GPU metrics, power, temperature,
 * and other system management information.
 */

#ifndef SMI_WORKER_HANDLERS_H
#define SMI_WORKER_HANDLERS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the SMI subsystem.
 * Called once when the worker starts.
 * @return 0 on success, negative on failure
 */
int smi_worker_init(void);

/**
 * Shutdown the SMI subsystem.
 * Called when the worker stops.
 */
void smi_worker_shutdown(void);

/**
 * Check if SMI is initialized and available.
 */
bool smi_worker_is_available(void);

/**
 * Get the number of GPU processors available.
 */
uint32_t smi_worker_get_processor_count(void);

/**
 * Dispatch an SMI operation.
 * @param client_fd Client socket file descriptor
 * @param op_code SMI operation code
 * @param request_id Request correlation ID
 * @param payload Request payload
 * @param payload_size Payload size in bytes
 * @return 0 on success, negative on failure
 */
int smi_worker_dispatch(
    int client_fd,
    uint16_t op_code,
    uint32_t request_id,
    const void* payload,
    size_t payload_size
);

#ifdef __cplusplus
}
#endif

#endif /* SMI_WORKER_HANDLERS_H */
