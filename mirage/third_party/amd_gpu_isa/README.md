# AMD GPU ISA Sources

Mirage vendors the official machine-readable AMD ISA XML used to generate
instruction catalogs for simulator bring-up.

Current source:

- `amdgpu_isa_cdna4.xml` for `gfx950` / AMD CDNA 4

Refresh workflow:

```bash
curl -L https://gpuopen.com/download/machine-readable-isa/latest/ \
  -o .cache/amd-isa/AMD_GPU_MR_ISA_XML_latest.zip
unzip -p .cache/amd-isa/AMD_GPU_MR_ISA_XML_latest.zip amdgpu_isa_cdna4.xml \
  > third_party/amd_gpu_isa/amdgpu_isa_cdna4.xml
python tools/generate_gfx950_isa_catalog.py
```

The machine-readable XML advertises `MIT` in the `Document/License` field and
is used here as the checked-in source of truth for generated metadata.
