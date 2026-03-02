"""RDNA4 (gfx120x) GPU family - RX 9000 series."""

from amd_gpu_driver.gpu.family import GPUFamilyConfig, register_gpu_family

_RDNA4_COMMON = dict(
    architecture="RDNA4",
    sh_reg_base=0x2C00,
    sh_reg_end=0x3000,
    uconfig_reg_base=0xC000,
    uconfig_reg_end=0x10000,
    sdma_major=7,
    sdma_minor=0,
    eop_buffer_size=4096,
    ctx_save_restore_size=0x2C02000,
    wave_size=32,
    max_workgroup_size=1024,
)

_RDNA4_TARGETS = [
    ("gfx1200", (12, 0, 0), 2),  # RX 9060/XT
    ("gfx1201", (12, 0, 1), 2),  # RX 9070/XT
]

for name, version, num_se in _RDNA4_TARGETS:
    register_gpu_family(
        GPUFamilyConfig(
            name=name,
            gfx_version=version,
            num_shader_engines=num_se,
            **_RDNA4_COMMON,
        )
    )
