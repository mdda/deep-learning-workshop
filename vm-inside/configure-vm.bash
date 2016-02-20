#!/bin/bash -

# This runs inside the guest at first boot.

set -e
set -x

cd /home/user
source configure-vm.conf

# Build from SRPM.
#rpmbuild --define '_topdir /home/build' --rebuild /home/build/$srpm

# If we get this far, everything installed/initialised successfully.
# This string is detected in the guest afterwards.
echo '=== CONFIGURATION FINISHED OK ==='
