#!/bin/bash -

set -e
set -x

source ./config/params

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
  -net user,hostfwd=tcp::$port_jupyter-:$port_jupyter,hostfwd=tcp::$port_tensorboard-:$port_tensorboard  \
  -net nic \
  -serial stdio \
  -drive file=$image_file,format=raw,if=virtio,cache=unsafe



#  http://serverfault.com/questions/704294/qemu-multiple-port-forwarding


# ,hostfwd=tcp::$port_tensorboard:$port_tensorboard
#  -net user,hostfwd=tcp::$port_jupyter-:$port_jupyter 
#  -net nic 
#  -net user,hostfwd=tcp::$port_jupyter-:$port_jupyter
