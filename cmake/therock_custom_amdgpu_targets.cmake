# gfx115X family
therock_add_amdgpu_target(gfx1154 "AMD gfx1154 IGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rccl # https://github.com/ROCm/TheRock/issues/150
    rccl-tests
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
