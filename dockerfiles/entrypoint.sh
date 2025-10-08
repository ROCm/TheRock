#!/bin/bash

if [ "$(id -u)" = "0" ]; then
  chown -R tester:tester /__w || true
  exec su -s /bin/bash tester -c "$*"
else
  exec "$@"
fi
