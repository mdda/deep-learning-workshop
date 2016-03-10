#!/bin/bash -

set -e
set -x

source ./vm-config/params

## See : https://developer.fedoraproject.org/tools/virt-builder/about.html

## Required for virt-builder ::
# dnf install libguestfs libguestfs-tools libvirt-daemon-config-network
# ->> libguestfs-1.30.6-2.fc22.x86_64 ...
# ->> without the libvirt-daemon-config-network, the bridging network thing doesn't get set up properly


## Running this script the first time takes an extra ~10mins to download 
##   http://libguestfs.org/download/builder/fedora-23.xz


# virt-builder: error: libguestfs error: bridge 'virbr0' not found.  Try 
# running:  brctl show

# If libvirt is being used then the appliance will connect to "virbr0"
#  (can be overridden by setting "LIBGUESTFS_BACKEND_SETTINGS=network_bridge=<some_bridge>").  
#  This enables full-featured network connections, with working ICMP, ping and so on.
#     ->> suggests that libvirt is not cooperating, somehow

## Try : LIBGUESTFS_BACKEND=direct
# LIBGUESTFS_BACKEND=direct; export LIBGUESTFS_BACKEND

## Easier to use libvirt properly (doesn't require LIBGUESTFS_BACKEND setting above), by configuring virtual networks using virtsh :

# virsh net-list --all
# virsh net-define /usr/share/libvirt/networks/default.xml
# virsh net-autostart default
# virsh net-start default
# brctl show




# The build script.
#build_script=/tmp/build-it.sh
# Because virt-builder copies the build script permissions too.
#chmod +x $build_script

# target location :: <REPO>/vm-images/


# How much guest memory we need for the build:
export LIBGUESTFS_MEMSIZE=4096

# Run virt-builder.
# Use a long build path to work around RHBZ#757089.
# $d/run $d/builder/virt-builder 


# virt-builder: error: images cannot be shrunk, the output size is too small 
# for this image.  Requested size = 4.0G, minimum size = 6.0G

## Filesystem with fedora-23 installed is actually mostly empty :
# Filesystem      Size  Used Avail Use% Mounted on
# devtmpfs        994M     0  994M   0% /dev
# tmpfs          1001M     0 1001M   0% /dev/shm
# tmpfs          1001M  216K 1001M   1% /run
# tmpfs          1001M     0 1001M   0% /sys/fs/cgroup
# /dev/vda3       5.0G  751M  4.2G  15% /
# tmpfs          1001M  4.0K 1001M   1% /tmp
# /dev/vda1       477M   74M  374M  17% /boot
# tmpfs           201M     0  201M   0% /run/user/1000

# Ensure that the python env cache exists
mkdir -p vm-guest/cache/env


virt-builder \
  $guest_type \
  --output $image_file \
  --root-password file:./vm-config/root-password \
  --commands-from-file vm-config/0-init \
  --commands-from-file vm-config/1-packages \
  --commands-from-file vm-config/3-user \
  --write "/home/user/configure-vm.conf:port=$port"
  
#  --firstboot-command 'poweroff'

#  --install /usr/bin/yum-builddep,/usr/bin/rpmbuild,@buildsys-build,@development-tools 
#  --run-command "yum-builddep -y /home/build/$srpm" 

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

# The build ran OK if this contains the magic string (see the build script ./vm-guest/configure-vm.bash) :
virt-cat -a $image_file /root/virt-sysprep-firstboot.log

# Copy out the SRPMs & RPMs.
#rm -rf /tmp/result
#mkdir /tmp/result
#virt-copy-out -a $image_file /home/build/RPMS /home/build/SRPMS /tmp/result

# Copy out the ~/env directory - so that it's cached between builds
#rm -rf ./vm-guest/cache/env
#virt-copy-out -a $image_file /home/env ./vm-guest/cache/

# Leave the guest around so you can examine the /home/build dir if you want.
# Or you could delete it.
#rm $image_file
