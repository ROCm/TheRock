# gfx1201 Phase-0 Wave32 Boundary

This report captures the current wave32-local boundary for the `gfx1201` phase-0
execution path.

## Saturated Local Encodings

- `ENC_VOP1`: seeded `90`, executable `90`, fully executable `true`
- `ENC_VOP2`: seeded `47`, executable `47`, fully executable `true`
- `ENC_VOPC`: seeded `162`, executable `162`, fully executable `true`

## Summary

- Phase-0 executable opcodes: `325`
- Wave size: `32`
- All currently seeded `ENC_VOP1`, `ENC_VOP2`, and `ENC_VOPC` instruction/encoding pairs are executable on the local path.
- There are no remaining imported `ENC_VOP1`, `ENC_VOP2`, or `ENC_VOPC` instruction/encoding pairs outside the current seed surface.
- Remaining narrow `ENC_VOP1`/`ENC_VOP2`/`ENC_VOPC` instruction/encoding pairs outside the current seed: `0`
- Recommended next frontier: `ENC_SMEM`

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

- `ENC_SMEM`: example `S_LOAD_B32`, seeded `28`, as-is `0`, decoder-rollup `3`, semantic-only `0`, gfx1201-specific `25`
- `ENC_VOP3`: example `V_ADD3_U32`, seeded `434`, as-is `232`, decoder-rollup `91`, semantic-only `24`, gfx1201-specific `87`
- `ENC_VDS`: example `DS_ADD_U32`, seeded `123`, as-is `27`, decoder-rollup `38`, semantic-only `0`, gfx1201-specific `58`
- `ENC_VGLOBAL`: example `GLOBAL_LOAD_B32`, seeded `65`, as-is `3`, decoder-rollup `0`, semantic-only `0`, gfx1201-specific `62`

`ENC_SMEM` is the recommended next frontier because it is the smallest remaining
seeded blocker and keeps the next phase architecture-local. The later frontier
steps move into broader decoder/execution churn, with `ENC_VOP3` remaining the
largest and riskiest step.
