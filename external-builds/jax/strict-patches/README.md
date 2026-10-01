# Native MI45x integration overlay

- JAX base: ROCm/jax `7353a2f071e7935f720358a757b7e539f47a2dd1` (0.11.2).
- rocm-jax build infrastructure: `431083f5809c124b5acb4298d87ae688179444f6`.
- JAX #40877: exact upstream diff, merged as `7e16149629055975b4862be7a544bdbe59fdd23e`, absent from selected downstream branch.
- XLA base: ROCm/xla `40613bb4aea1f0cad475da79c570cd394563e83e` from JAX MODULE.bazel.
- XLA #49806 head: `5a06c6676f02493d5d4c303ee055ef30385c2073` (open).

XLA patch rebases the upstream change onto this downstream pin: preserve the
existing `gfx1250` test target naming (upstream renamed it `mi450`), insert the
strict peer, preserve downstream GPU list membership and supported-target list
format. No unrelated upstream changes are imported. The added target spec still
contains upstream placeholder hardware values and is not physical qualification.

Bzlmod applies the patch to the existing integrity-pinned ROCm XLA archive.
JAX target list is retained from the upstream patch, including older architectures;
`device-all` selects packages available in the run index. Successful compilation
and complete target coverage remain CI evidence to collect, not an assertion here.
