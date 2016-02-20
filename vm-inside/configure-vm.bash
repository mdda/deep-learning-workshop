#!/bin/bash -

# This runs inside the throwaway guest at first boot.

set -e
set -x

cd /home/user
#source config

# Build from SRPM.
#rpmbuild --define '_topdir /home/build' --rebuild /home/build/$srpm

# If we get this far, everything built successfully.
# This string is detected in the guest afterwards.
echo '=== BUILD FINISHED OK ==='
