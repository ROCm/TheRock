#!/bin/bash

td="$(cd $(dirname $0) && pwd)"
pp="$(cd $td/.. && pwd)"

python -m build $pp/rocm-sdk -v --sdist
python -m build $pp/rocm-sdk-core -v --wheel
