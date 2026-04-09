---
author: Liam Berry (LiamfBerry), Saad Rahim (saadrahim)
created: 2026-04-09
modified: 2026-04-09
status: draft
---

### ZIP Package Requirements

ZIP packages must provide a portable file-tree representation of a Windows ROCm installation.

Loose Files Package (ZIP): ZIP archive containing a directory tree identical to the MSI-installed layout, intended for power users, CI, or offline deployment

ZIP archive layout must match the labelled directory layout:

```
rocm-core-X.Y.Z.zip
  rocm-core-X.Y\bin\...
```

ZIP packages:

- Must not modify environment variables
- Must not modify `PATH`
- Must not create registry entries
- Must remain suitable for CI, offline deployment, and advanced users

Tools and scripts inside ZIP packages should function correctly when the extracted directory is used directly as an SDK root.

AMD Official Repository (repo.amd.com/rocm/windows) hosts ZIP archives for Winget ingestion or internal automation.
