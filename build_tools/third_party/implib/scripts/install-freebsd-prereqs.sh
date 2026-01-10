#!/bin/sh

set -eu

pkg update
pkg upgrade -y
pkg install -y gcc binutils python3
