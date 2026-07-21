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
 * @file smi-remote.c
 * @brief Remote AMD SMI command-line tool
 *
 * Usage:
 *   smi-remote [--host HOST] [--port PORT] COMMAND [ARGS]
 *
 * Commands:
 *   list      - List available GPUs
 *   metrics   - Show GPU metrics (all GPUs or specific index)
 *   power     - Show power information
 *   info      - Show ASIC information
 *
 * Examples:
 *   smi-remote --host sharkmi300x list
 *   smi-remote --host sharkmi300x metrics
 *   smi-remote --host sharkmi300x metrics 0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "smi_remote/smi_remote_client.h"

#ifdef _WIN32
#define hip_cli_setenv(name, value) _putenv_s(name, value)
#else
#define hip_cli_setenv(name, value) setenv(name, value, 1)
#endif

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [OPTIONS] COMMAND [ARGS]\n\n", prog);
    fprintf(stderr, "Remote AMD SMI query tool\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --host HOST    Worker hostname (or TF_WORKER_HOST env)\n");
    fprintf(stderr, "  --port PORT    Worker port (default: 18515)\n");
    fprintf(stderr, "  -h, --help     Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Commands:\n");
    fprintf(stderr, "  list           List available GPUs\n");
    fprintf(stderr, "  metrics [N]    Show GPU metrics (all or GPU N)\n");
    fprintf(stderr, "  power [N]      Show power information\n");
    fprintf(stderr, "  info [N]       Show ASIC information\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s --host myserver list\n", prog);
    fprintf(stderr, "  %s --host myserver metrics 0\n", prog);
    fprintf(stderr, "\n");
}

static int cmd_list(void) {
    smi_remote_status_t status = smi_remote_init();
    if (status != SMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to initialize: %s\n", smi_remote_status_string(status));
        return 1;
    }

    uint32_t count = 0;
    status = smi_remote_get_processor_count(&count);
    if (status != SMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to get processor count: %s\n", smi_remote_status_string(status));
        return 1;
    }

    printf("GPU Count: %u\n\n", count);

    for (uint32_t i = 0; i < count; i++) {
        smi_remote_asic_info_t info;
        memset(&info, 0, sizeof(info));
        status = smi_remote_get_asic_info(i, &info);
        if (status == SMI_STATUS_SUCCESS) {
            printf("GPU %u: %s\n", i, info.market_name);
            printf("  Vendor ID:  0x%04X\n", info.vendor_id);
            printf("  Device ID:  0x%04X\n", info.device_id);
            printf("  Compute Units: %u\n", info.num_compute_units);
            if (info.serial[0]) {
                printf("  Serial: %s\n", info.serial);
            }
            printf("\n");
        } else {
            printf("GPU %u: (failed to get info)\n\n", i);
        }
    }

    return 0;
}

static void print_metrics_header(void) {
    printf("%-4s %-24s %6s %6s %6s %6s %6s %8s %8s %10s\n",
           "GPU", "Name", "Temp", "Power", "GFX%", "Mem%", "GFXClk", "MemClk", "VRAMUsed", "VRAMTotal");
    printf("%-4s %-24s %6s %6s %6s %6s %6s %8s %8s %10s\n",
           "---", "------------------------", "------", "------", "------", "------", "------", "--------", "--------", "----------");
}

static int cmd_metrics(int gpu_index) {
    smi_remote_status_t status = smi_remote_init();
    if (status != SMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to initialize: %s\n", smi_remote_status_string(status));
        return 1;
    }

    uint32_t count = 0;
    status = smi_remote_get_processor_count(&count);
    if (status != SMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to get processor count: %s\n", smi_remote_status_string(status));
        return 1;
    }

    if (gpu_index >= 0 && (uint32_t)gpu_index >= count) {
        fprintf(stderr, "GPU %d not found (have %u GPUs)\n", gpu_index, count);
        return 1;
    }

    uint32_t start = (gpu_index >= 0) ? (uint32_t)gpu_index : 0;
    uint32_t end = (gpu_index >= 0) ? (uint32_t)(gpu_index + 1) : count;

    print_metrics_header();

    for (uint32_t i = start; i < end; i++) {
        smi_remote_asic_info_t info;
        smi_remote_gpu_metrics_t metrics;
        memset(&info, 0, sizeof(info));
        memset(&metrics, 0, sizeof(metrics));

        smi_remote_get_asic_info(i, &info);
        status = smi_remote_get_gpu_metrics(i, &metrics);

        if (status == SMI_STATUS_SUCCESS) {
            char name[25];
            strncpy(name, info.market_name, 24);
            name[24] = '\0';

            double vram_used_gb = metrics.vram_used_bytes / (1024.0 * 1024.0 * 1024.0);
            double vram_total_gb = metrics.vram_total_bytes / (1024.0 * 1024.0 * 1024.0);

            printf("%-4u %-24s %5dC %5uW %5u%% %5u%% %5uMHz %6uMHz %7.1fGB %8.1fGB\n",
                   i, name,
                   metrics.temperature_hotspot,
                   metrics.power_watts,
                   metrics.gfx_activity,
                   metrics.mem_activity,
                   metrics.gfx_clock_mhz,
                   metrics.mem_clock_mhz,
                   vram_used_gb,
                   vram_total_gb);
        } else {
            printf("%-4u (failed to get metrics)\n", i);
        }
    }

    return 0;
}

static int cmd_power(int gpu_index) {
    smi_remote_status_t status = smi_remote_init();
    if (status != SMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to initialize: %s\n", smi_remote_status_string(status));
        return 1;
    }

    uint32_t count = 0;
    status = smi_remote_get_processor_count(&count);
    if (status != SMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to get processor count: %s\n", smi_remote_status_string(status));
        return 1;
    }

    if (gpu_index >= 0 && (uint32_t)gpu_index >= count) {
        fprintf(stderr, "GPU %d not found (have %u GPUs)\n", gpu_index, count);
        return 1;
    }

    uint32_t start = (gpu_index >= 0) ? (uint32_t)gpu_index : 0;
    uint32_t end = (gpu_index >= 0) ? (uint32_t)(gpu_index + 1) : count;

    for (uint32_t i = start; i < end; i++) {
        smi_remote_asic_info_t info;
        smi_remote_power_info_t power;
        memset(&info, 0, sizeof(info));
        memset(&power, 0, sizeof(power));

        smi_remote_get_asic_info(i, &info);
        status = smi_remote_get_power_info(i, &power);

        printf("GPU %u: %s\n", i, info.market_name);
        if (status == SMI_STATUS_SUCCESS) {
            printf("  Current Power:  %u W\n", power.current_power_watts);
            printf("  Average Power:  %u W\n", power.average_power_watts);
            printf("  Power Limit:    %u W\n", power.power_limit_watts);
            printf("  GFX Voltage:    %u mV\n", power.gfx_voltage_mv);
            printf("  SOC Voltage:    %u mV\n", power.soc_voltage_mv);
            printf("  Mem Voltage:    %u mV\n", power.mem_voltage_mv);
        } else {
            printf("  (failed to get power info)\n");
        }
        printf("\n");
    }

    return 0;
}

static int cmd_info(int gpu_index) {
    smi_remote_status_t status = smi_remote_init();
    if (status != SMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to initialize: %s\n", smi_remote_status_string(status));
        return 1;
    }

    uint32_t count = 0;
    status = smi_remote_get_processor_count(&count);
    if (status != SMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to get processor count: %s\n", smi_remote_status_string(status));
        return 0;
    }

    if (gpu_index >= 0 && (uint32_t)gpu_index >= count) {
        fprintf(stderr, "GPU %d not found (have %u GPUs)\n", gpu_index, count);
        return 1;
    }

    uint32_t start = (gpu_index >= 0) ? (uint32_t)gpu_index : 0;
    uint32_t end = (gpu_index >= 0) ? (uint32_t)(gpu_index + 1) : count;

    for (uint32_t i = start; i < end; i++) {
        smi_remote_asic_info_t info;
        memset(&info, 0, sizeof(info));
        status = smi_remote_get_asic_info(i, &info);

        printf("GPU %u:\n", i);
        if (status == SMI_STATUS_SUCCESS) {
            printf("  Name:           %s\n", info.market_name);
            printf("  Vendor ID:      0x%04X\n", info.vendor_id);
            printf("  Device ID:      0x%04X\n", info.device_id);
            printf("  Rev ID:         0x%02X\n", info.rev_id);
            printf("  Compute Units:  %u\n", info.num_compute_units);
            if (info.serial[0]) {
                printf("  Serial:         %s\n", info.serial);
            }
        } else {
            printf("  (failed to get info)\n");
        }
        printf("\n");
    }

    return 0;
}

int main(int argc, char** argv) {
    const char* command = NULL;
    int gpu_index = -1;  /* -1 means all GPUs */
    int argi = 1;

    /* Parse options */
    while (argi < argc) {
        if (strcmp(argv[argi], "--host") == 0 && argi + 1 < argc) {
            hip_cli_setenv("TF_WORKER_HOST", argv[argi + 1]);
            argi += 2;
        } else if (strcmp(argv[argi], "--port") == 0 && argi + 1 < argc) {
            hip_cli_setenv("TF_WORKER_PORT", argv[argi + 1]);
            argi += 2;
        } else if (strcmp(argv[argi], "-h") == 0 || strcmp(argv[argi], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (argv[argi][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n\n", argv[argi]);
            print_usage(argv[0]);
            return 1;
        } else {
            /* First non-option argument is the command */
            command = argv[argi];
            argi++;
            /* Second non-option argument (if any) is GPU index */
            if (argi < argc) {
                gpu_index = atoi(argv[argi]);
            }
            break;
        }
    }

    if (!command) {
        fprintf(stderr, "Error: No command specified\n\n");
        print_usage(argv[0]);
        return 1;
    }

    int result = 0;

    if (strcmp(command, "list") == 0) {
        result = cmd_list();
    } else if (strcmp(command, "metrics") == 0) {
        result = cmd_metrics(gpu_index);
    } else if (strcmp(command, "power") == 0) {
        result = cmd_power(gpu_index);
    } else if (strcmp(command, "info") == 0) {
        result = cmd_info(gpu_index);
    } else {
        fprintf(stderr, "Unknown command: %s\n\n", command);
        print_usage(argv[0]);
        result = 1;
    }

    smi_remote_shutdown();
    return result;
}
