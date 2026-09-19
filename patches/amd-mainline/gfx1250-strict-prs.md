# gfx1250-strict component PR integration

This integration branch uses component PR commits instead of the previous custom SDK patch stack. Snapshot: 2026-09-19.

| Repository | PR | Source head |
| --- | --- | --- |
| rocm-libraries | [#11620](https://github.com/ROCm/rocm-libraries/pull/11620) | `1af0df63a4042563b4301303a97d9939a302a153` |
| rocm-libraries | [#11777](https://github.com/ROCm/rocm-libraries/pull/11777) | `43cdd0e355f1c0a194fa460a9337ad786507ea74` |
| rocm-libraries | [#10052](https://github.com/ROCm/rocm-libraries/pull/10052) | `bf56af2e53b7ba22b4ac86e8d8cd25df8040b0f2` |
| rocm-libraries | [#12069](https://github.com/ROCm/rocm-libraries/pull/12069) | `e7869bdae3ff03ecc268efb0f1727dc26aef44cf` |
| rocm-libraries | [#12321](https://github.com/ROCm/rocm-libraries/pull/12321) | `1ecae4c3a5a8680d7dffa7aa81532da35206bfb4` |
| rocm-libraries | [#12203](https://github.com/ROCm/rocm-libraries/pull/12203) | `f0c0a3c85f3650ed739834c46289581ed36e9efa` |
| rocm-systems | [#11575](https://github.com/ROCm/rocm-systems/pull/11575) | `39b36bce064595e1d84a90b702939334c00aee3a` |
| rocm-systems | [#11526](https://github.com/ROCm/rocm-systems/pull/11526) | `6c90bdc78f0a4f698e07b64b3794d37ba2f2b970` |
| rocm-systems | [#11639](https://github.com/ROCm/rocm-systems/pull/11639) | `a22ff4ba34dc0e94ca22ea7f42a9e6c5f89a0e1a` |
| rocm-systems | [#11737](https://github.com/ROCm/rocm-systems/pull/11737) | `056ef984360b80bef95c66abff9d0a1fb0a5ef73` |
| pytorch | [#3639](https://github.com/ROCm/pytorch/pull/3639) | `5c17a5d2a87be40cf3c38b033d6584f462c928c3` |
| xla | [#1208](https://github.com/ROCm/xla/pull/1208) | `6f15d53df38a14173b7dbf55494ee3438ee2bf71` |

## Source selection and application

- TheRock #8350 is deferred. Its bootstrap/kpack prerequisite rocm-systems #11801 is now inherited from mainline. Explicit selection of both targets remains available. Component SDK commits are author-preserving patches in lexicographic application order, with upstream commit references.
- Library base: `a4bfdbfed4f128286ea0af1c0d4f224e1b35bfeb`; systems base: `66e84e2576c6e755a9bb6ce20a57413ebbeb83d0`; compiler base remains `6bd80f15ed27c64da2da6f8fdffdbd2621f32f18`.
- hipBLASLt PR #11777 uses the merged true16 fix from #10052, now included in the library pin; the duplicate patch is removed. Four YAML marker conflicts retain the pinned source's existing markers and add the PR's strict skip marker; unrelated ordinary-gfx1250 markers from a different base were not introduced.
- PyTorch Linux builds use the exact #3639 head on its release/2.13 source base. JAX Linux builds select rocm-jaxlib-v0.11.1 and override XLA with the exact #1208 head, including that PR's own base and LLVM dependency pin. Older framework release matrices are not backported on this branch.
- The previous custom RCCL patch is removed. No matching dedicated open strict-target PR was found for RCCL, rocRoller, mxDataGenerator, or Triton in this audit. This is not a claim that those components work without changes.
- PR #12327 is excluded because its default skips strict Tensile generation. Already-pinned merged component changes are not duplicated.

## Verification

All 35 library patches and 4 systems patches replay successfully onto their exact pins. TheRock packaging, target, artifact, CI configuration, and framework matrix tests pass (232 passed, 1 skipped, 49 subtests passed). The selected JAX build script supports the local XLA override. These are source integration checks, not evidence of a successful full SDK or framework wheel build. No workflow was launched as part of this refresh.
