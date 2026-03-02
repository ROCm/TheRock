"""CDNA2 (gfx90a) GPU family - MI210/MI250."""

from amd_gpu_driver.gpu.family import GPUFamilyConfig, register_gpu_family

GFX90A = GPUFamilyConfig(
    name="gfx90a",
    architecture="CDNA2",
    gfx_version=(9, 0, 10),
    sh_reg_base=0x2C00,
    sh_reg_end=0x3000,
    uconfig_reg_base=0xC000,
    uconfig_reg_end=0x10000,
    sdma_major=4,
    sdma_minor=4,
    eop_buffer_size=4096,
    ctx_save_restore_size=0x2C02000,
    wave_size=64,
    num_shader_engines=4,
    max_workgroup_size=1024,
)

register_gpu_family(GFX90A)
