#!/bin/bash -

set -e
set -x

source ./vm-config/params

# Run qemu directly.  Could also use virt-install --import here.

# Documentation : http://dev.man-online.org/man1/qemu-system-x86_64/
qemu-system-x86_64 \
  -nodefconfig \
  -nodefaults \
  -nographic \
  -machine accel=kvm:tcg \
  -cpu host \
  -m 2048 \
  -smp 4 \
  -net nic -net user \
  -serial stdio \
  -drive file=$image_file,format=raw,if=virtio,cache=unsafe
