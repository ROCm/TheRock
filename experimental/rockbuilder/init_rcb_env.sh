#!/bin/bash

# Check that this script is called as sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced, not executed!" >&2
    echo "Example:" >&2
    echo "source ./init_rcb_env.sh" >&2
    exit 1
fi

if [ -f ./.venv/bin/activate ]; then
	source .venv/bin/activate
else
	python3 -m venv .venv && source .venv/bin/activate
	pip install --upgrade pip
	pip install -r ../../requirements.txt
fi
