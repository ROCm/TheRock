# gfx1201 Phase-0 Wave32 Boundary

This report captures the current wave32-local boundary for the `gfx1201` phase-0
execution path.

## Saturated Local Encodings

- `ENC_VOP1`: seeded `90`, executable `90`, fully executable `true`
- `ENC_VOP2`: seeded `47`, executable `47`, fully executable `true`
- `ENC_VOPC`: seeded `162`, executable `162`, fully executable `true`
- `ENC_SMEM`: seeded `28`, executable `28`, fully executable `true`

## Summary

- Phase-0 executable opcodes: `378`
- Wave size: `32`
- All currently seeded `ENC_VOP1`, `ENC_VOP2`, `ENC_VOPC`, and `ENC_SMEM` instruction/encoding pairs are executable on the local path.
- There are no remaining imported `ENC_VOP1`, `ENC_VOP2`, `ENC_VOPC`, or `ENC_SMEM` instruction/encoding pairs outside the current seed surface.
- `ENC_SMEM` is now saturated through `S_DCACHE_INV`, the prefetch and ATC-probe footholds, the full non-buffer `S_LOAD_*` slice, and the matching `S_BUFFER_LOAD_*` slice, leaving no seeded scalar-memory instructions scaffolded.
- Remaining narrow `ENC_VOP1`/`ENC_VOP2`/`ENC_VOPC` instruction/encoding pairs outside the current seed: `0`
- Recommended next frontier: `ENC_VGLOBAL`

## Next-Risk Encodings

- `ENC_SMEM`
- `ENC_VOP3`
- `ENC_VDS`
- `ENC_VGLOBAL`

## Suggested Frontier Order

- `ENC_SMEM`
- `ENC_VGLOBAL`
- `ENC_VDS`
- `ENC_VOP3`

## Next-Risk Encoding Status

- `ENC_SMEM`: example `S_LOAD_B32`, seeded `28`, executable `28` via `S_ATC_PROBE`, as-is `0`, decoder-rollup `3`, semantic-only `0`, gfx1201-specific `25`
- `ENC_VOP3`: example `V_ADD3_U32`, seeded `434`, executable `0`, as-is `232`, decoder-rollup `91`, semantic-only `24`, gfx1201-specific `87`
- `ENC_VDS`: example `DS_ADD_U32`, seeded `123`, executable `0`, as-is `27`, decoder-rollup `38`, semantic-only `0`, gfx1201-specific `58`
- `ENC_VGLOBAL`: example `GLOBAL_LOAD_B32`, seeded `65`, executable `25` via `GLOBAL_INV`, as-is `3`, decoder-rollup `0`, semantic-only `0`, gfx1201-specific `62`

`ENC_VGLOBAL` remains the recommended next frontier because `ENC_SMEM` is saturated,
and the current local footholds now include `GLOBAL_INV`, `GLOBAL_WB`, `GLOBAL_WBINV`,
the plain `GLOBAL_LOAD_U8`/`I8`/`U16`/`I16`/`B32`/`B64`/`B96`/`B128` slice, the packed
`GLOBAL_LOAD_D16_U8`/`I8`/`B16` and `GLOBAL_LOAD_D16_HI_U8`/`I8`/`B16` slice, the plain
`GLOBAL_STORE_B8`/`B16`/`B32`/`B64`/`B96`/`B128` slice, and `GLOBAL_STORE_D16_HI_B8`/`B16`
without committing to broader vector-memory `TR`, `BLOCK`, `ADDTID`, or atomic semantics.
The later frontier steps move into broader
decoder/execution churn, with `ENC_VOP3` remaining the largest and riskiest
step.
