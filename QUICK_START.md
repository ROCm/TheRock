***IMPORTANT: MAKE SURE NO AMD ROCm COMPILER TOOLS ARE IN YOUR `PATH`
   OR `LD_LIBRARY_PATH` ENVIRONMENT VARIABLES***

In the cmake command below, change `gfx1100` to whatever GPU family or
families you're interested in.  Separate with commas, I think, or
semicolons if that doesn't work.

#If you don't want ccache, delete everything after
 `-DTHEROCK_ENABLE_MATH_LIBS=ON` in the cmake command and don't bother
 running `eval "$(./build_tools/setup_ccache.py)"`.

```bash
git clone https://github.com/ROCm/TheRock.git
cd TheRock
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python ./build_tools/fetch_sources.py
eval "$(./build_tools/setup_ccache.py)"
cmake -GNinja -B build -DTHEROCK_AMDGPU_FAMILIES=gfx1100 -DTHEROCK_ENABLE_ALL=OFF -DTHEROCK_ENABLE_COMPILER=ON  -DTHEROCK_ENABLE_CORE_RUNTIME=ON -DTHEROCK_ENABLE_HIP_RUNTIME=ON -DTHEROCK_ENABLE_RCCL=ON -DTHEROCK_ENABLE_PRIM=ON -DTHEROCK_ENABLE_BLAS=ON -DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_SOLVER=ON -DTHEROCK_ENABLE_SPARSE=ON -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF -DTHEROCK_ENABLE_OCL_RUNTIME=ON  -DTHEROCK_ENABLE_MATH_LIBS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cd build
ninja
```

Easily Reproducing CI Failures
------------------------------

To debug a CI failure, the quick and dirty way is to switch your
`llvm-project` checkout in the `compiler/amd-llvm` directory to your
PR's branch.  Then, follow the instructions above, making sure to set
`-DTHEROCK_AMDGPU_FAMILIES` to the same GPU family as the family that
failed in the CI.  Check the logs for the CI for the failing command
since it will have the correct GPU family.  It should not be necessary
to use a Docker instance, but you can get the Docker from the CI logs
if you want it.  The line you need looks like this:

```
2026-02-07T20:07:46.1202126Z ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:6e8242d347af7e0c43c82d5031a3ac67b669f24898ea8dc2f1d5b7e4798b66bd: Pulling from rocm/therock_build_manylinux_x86_64``
```

Reproducing CI Failures the Hard Way
------------------------------------

One thing to watch out for, which may or may not cause a problem, is
that the CI does autopatch `llvm-project` by putting a few commits on
top of your PR's branch.  To handle this, run the command:
```
git update-index --cacheinfo 160000,$PR_SHA,compiler/amd-llvm
```

Replacing $PR_SHA with your PR's SHA.  Then, re-run:
```
python ./build_tools/fetch_sources.py
```
