#!/bin/bash

# runner setup
mkdir "actions-runner-$1" && cd "actions-runner-$1"
curl -o "actions-runner-linux-x64-$2.tar.gz" -L https://github.com/actions/runner/releases/download/v$2/actions-runner-linux-x64-$2.tar.gz
tar xzf ./actions-runner-linux-x64-$2.tar.gz
