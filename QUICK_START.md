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
