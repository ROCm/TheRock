"""CDNA3 (gfx942) GPU family - MI300A/MI300X."""

from amd_gpu_driver.gpu.family import GPUFamilyConfig, register_gpu_family

GFX942 = GPUFamilyConfig(
    name="gfx942",
    architecture="CDNA3",
    gfx_version=(9, 4, 2),
    sh_reg_base=0x2C00,
    sh_reg_end=0x3000,
    uconfig_reg_base=0xC000,
    uconfig_reg_end=0x10000,
    sdma_major=4,
    sdma_minor=4,
    eop_buffer_size=4096,
    ctx_save_restore_size=0x2C02000,
    wave_size=64,
    num_shader_engines=8,
    max_workgroup_size=1024,
)

register_gpu_family(GFX942)
