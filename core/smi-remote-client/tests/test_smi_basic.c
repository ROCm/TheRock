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
 * @file test_smi_basic.c
 * @brief Basic tests for remote SMI client
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "smi_remote/smi_remote_client.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, condition) do { \
    if (condition) { \
        printf("  PASS: %s\n", name); \
        tests_passed++; \
    } else { \
        printf("  FAIL: %s\n", name); \
        tests_failed++; \
    } \
} while(0)

static void test_init(void) {
    printf("\nTest: smi_remote_init\n");
    smi_remote_status_t status = smi_remote_init();
    TEST("init succeeds", status == SMI_STATUS_SUCCESS);
    TEST("is connected", smi_remote_is_connected());
}

static void test_processor_count(void) {
    printf("\nTest: smi_remote_get_processor_count\n");
    uint32_t count = 0;
    smi_remote_status_t status = smi_remote_get_processor_count(&count);
    TEST("get count succeeds", status == SMI_STATUS_SUCCESS);
    TEST("count > 0", count > 0);
    printf("  Found %u GPU(s)\n", count);
}

static void test_gpu_metrics(void) {
    printf("\nTest: smi_remote_get_gpu_metrics\n");
    smi_remote_gpu_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));
    smi_remote_status_t status = smi_remote_get_gpu_metrics(0, &metrics);
    TEST("get metrics succeeds", status == SMI_STATUS_SUCCESS);
    TEST("temperature valid", metrics.temperature_hotspot >= 0 && metrics.temperature_hotspot < 150);
    printf("  Temperature: %d C\n", metrics.temperature_hotspot);
    printf("  Power: %u W\n", metrics.power_watts);
    printf("  GFX Activity: %u%%\n", metrics.gfx_activity);
    printf("  GFX Clock: %u MHz\n", metrics.gfx_clock_mhz);
    printf("  VRAM Used: %.1f GB / %.1f GB\n",
           metrics.vram_used_bytes / (1024.0 * 1024.0 * 1024.0),
           metrics.vram_total_bytes / (1024.0 * 1024.0 * 1024.0));
}

static void test_power_info(void) {
    printf("\nTest: smi_remote_get_power_info\n");
    smi_remote_power_info_t power;
    memset(&power, 0, sizeof(power));
    smi_remote_status_t status = smi_remote_get_power_info(0, &power);
    TEST("get power info succeeds", status == SMI_STATUS_SUCCESS);
    printf("  Current Power: %u W\n", power.current_power_watts);
    printf("  Power Limit: %u W\n", power.power_limit_watts);
}

static void test_asic_info(void) {
    printf("\nTest: smi_remote_get_asic_info\n");
    smi_remote_asic_info_t info;
    memset(&info, 0, sizeof(info));
    smi_remote_status_t status = smi_remote_get_asic_info(0, &info);
    TEST("get asic info succeeds", status == SMI_STATUS_SUCCESS);
    TEST("has market name", info.market_name[0] != '\0');
    printf("  Name: %s\n", info.market_name);
    printf("  Device ID: 0x%04X\n", info.device_id);
    printf("  Compute Units: %u\n", info.num_compute_units);
}

static void test_vram_usage(void) {
    printf("\nTest: smi_remote_get_vram_usage\n");
    uint64_t total = 0, used = 0;
    smi_remote_status_t status = smi_remote_get_vram_usage(0, &total, &used);
    TEST("get vram usage succeeds", status == SMI_STATUS_SUCCESS);
    TEST("total > 0", total > 0);
    TEST("used <= total", used <= total);
    printf("  VRAM: %.1f / %.1f GB\n",
           used / (1024.0 * 1024.0 * 1024.0),
           total / (1024.0 * 1024.0 * 1024.0));
}

static void test_gpu_activity(void) {
    printf("\nTest: smi_remote_get_gpu_activity\n");
    uint32_t gfx = 0, mem = 0, mm = 0;
    smi_remote_status_t status = smi_remote_get_gpu_activity(0, &gfx, &mem, &mm);
    TEST("get activity succeeds", status == SMI_STATUS_SUCCESS);
    TEST("gfx <= 100", gfx <= 100);
    TEST("mem <= 100", mem <= 100);
    printf("  GFX: %u%%, Mem: %u%%, MM: %u%%\n", gfx, mem, mm);
}

int main(void) {
    printf("=== Remote SMI Basic Tests ===\n");

    test_init();
    test_processor_count();
    test_gpu_metrics();
    test_power_info();
    test_asic_info();
    test_vram_usage();
    test_gpu_activity();

    smi_remote_shutdown();

    printf("\n=== Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
