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

## Next-Risk Encodings

- `ENC_SMEM`
- `ENC_VOP3`
- `ENC_VDS`
- `ENC_VGLOBAL`

The next coherent phase-0 extensions now move into those encodings, or require
broader shared-layer churn, rather than more narrow `VOP1`/`VOP2`/`VOPC` seed work.
