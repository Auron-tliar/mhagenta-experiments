#!/bin/sh
set -eu

apt-get update
apt-get install -y --no-install-recommends default-jre-headless
java -version
rm -rf /var/lib/apt/lists/*
