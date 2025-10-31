#!/usr/bin/env python3
"""
Fetch JAX wheels from a cloudfront url.
"""
import os, re, sys, fnmatch
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen, urlretrieve

wh = os.environ.get("WHEELHOUSE_DIR", "wheelhouse")
pv, pi, af = os.environ.get("PY_VER"), os.environ.get("PACKAGE_INDEX_URL"), os.environ.get("AMDGPU_FAMILY")
if not (pv and pi and af): sys.exit("Set PY_VER, PACKAGE_INDEX_URL, AMDGPU_FAMILY")

# Build the CP tag and the base listing URL
cp, base = pv.replace(".", ""), f"{pi.rstrip('/')}/{af.strip('/')}/"
os.makedirs(wh, exist_ok=True)

def links(u):
    """Fetch a page and return absolute HREFs found on it. On failure, return empty list."""
    try:
        h = urlopen(u, timeout=30).read().decode("utf-8", "ignore")
    except Exception:
        return []
    return [urljoin(u, x) for x in re.findall(r'href=[\'"]([^\'"]+)[\'"]', h, flags=re.I)]

# Collect links from base and from one level of subdirectories
lvl0 = links(base)
lvl1 = sum([links(u) for u in lvl0 if urlparse(u).path.endswith("/")], [])

# files: any .whl found at base or in its immediate subdirs
cands = []
for u in lvl0 + lvl1:
    p, f = urlparse(u).path, os.path.basename(urlparse(u).path)
    if f.endswith(".whl"):
        cands.append((f, u))

# Accept patterns equivalent to wget --accept filters
patterns = [
    f"jax_rocm7_plugin-*-cp{cp}-cp{cp}-manylinux_*_x86_64.whl",
    "jax_rocm7_pjrt-*-py3-none-manylinux_*_x86_64.whl",
]

# Filter candidates against patterns
hits = [(f, u) for f, u in cands if any(fnmatch.fnmatch(f, p) for p in patterns)]
if not hits:
    sys.exit("No matching wheels found.")

# Download wheels to wheelhouse (skip if already present)
for f, u in hits:
    dst = os.path.join(wh, f)
    if os.path.exists(dst):
        print("SKIP", f)
        continue
    urlretrieve(u, dst)
    print("OK", f)

# Print resulting contents
print("Wheelhouse:", *sorted(os.listdir(wh)), sep="\n  ")
