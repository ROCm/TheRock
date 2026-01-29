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
 * @file hip_api_stream.c
 * @brief Stream and event API implementations for remote HIP
 */

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"

/* ============================================================================
 * Stream Operations
 * ============================================================================ */

/* Stream handle type (opaque) */
typedef void* hipStream_t;
typedef void* hipEvent_t;

hipError_t hipStreamCreate(hipStream_t* stream) {
    if (!stream) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamCreateRequest req = {
        .flags = 0,
        .priority = 0
    };
    HipRemoteStreamCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_CREATE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *stream = (hipStream_t)(uintptr_t)resp.stream;
    } else {
        *stream = NULL;
    }
    return err;
}

hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    if (!stream) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamCreateRequest req = {
        .flags = flags,
        .priority = 0
    };
    HipRemoteStreamCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_CREATE_WITH_FLAGS,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *stream = (hipStream_t)(uintptr_t)resp.stream;
    } else {
        *stream = NULL;
    }
    return err;
}

hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags,
                                        int priority) {
    if (!stream) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamCreateRequest req = {
        .flags = flags,
        .priority = priority
    };
    HipRemoteStreamCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_CREATE_WITH_PRIORITY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *stream = (hipStream_t)(uintptr_t)resp.stream;
    } else {
        *stream = NULL;
    }
    return err;
}

hipError_t hipStreamDestroy(hipStream_t stream) {
    if (!stream) {
        return hipSuccess;  /* NULL stream is default stream, don't destroy */
    }

    HipRemoteStreamRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_DESTROY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
    HipRemoteStreamRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_SYNCHRONIZE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipStreamQuery(hipStream_t stream) {
    HipRemoteStreamRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_QUERY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                               unsigned int flags) {
    HipRemoteStreamWaitEventRequest req = {
        .stream = (uint64_t)(uintptr_t)stream,
        .event = (uint64_t)(uintptr_t)event,
        .flags = flags
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_WAIT_EVENT,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

/* ============================================================================
 * Event Operations
 * ============================================================================ */

hipError_t hipEventCreate(hipEvent_t* event) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventCreateRequest req = {
        .flags = 0
    };
    HipRemoteEventCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_EVENT_CREATE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *event = (hipEvent_t)(uintptr_t)resp.event;
    } else {
        *event = NULL;
    }
    return err;
}

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventCreateRequest req = {
        .flags = flags
    };
    HipRemoteEventCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_EVENT_CREATE_WITH_FLAGS,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *event = (hipEvent_t)(uintptr_t)resp.event;
    } else {
        *event = NULL;
    }
    return err;
}

hipError_t hipEventDestroy(hipEvent_t event) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventRequest req = {
        .event = (uint64_t)(uintptr_t)event
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_EVENT_DESTROY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventRecordRequest req = {
        .event = (uint64_t)(uintptr_t)event,
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_EVENT_RECORD,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipEventSynchronize(hipEvent_t event) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventRequest req = {
        .event = (uint64_t)(uintptr_t)event
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_EVENT_SYNCHRONIZE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipEventQuery(hipEvent_t event) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventRequest req = {
        .event = (uint64_t)(uintptr_t)event
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_EVENT_QUERY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    if (!ms || !start || !stop) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventElapsedTimeRequest req = {
        .start_event = (uint64_t)(uintptr_t)start,
        .end_event = (uint64_t)(uintptr_t)stop
    };
    HipRemoteEventElapsedTimeResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_EVENT_ELAPSED_TIME,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *ms = resp.milliseconds;
    }
    return err;
}
