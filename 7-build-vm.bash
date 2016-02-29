#!/bin/bash -

set -e
set -x

# Use a homebrew compile of libguestfs from git.
#d=$HOME/d/libguestfs

# See : https://developer.fedoraproject.org/tools/virt-builder/about.html

# This is the Fedora platform we want to build on.
guest_type=fedora-23

# The SRPM we want to build.
#srpmdir=/home/rjones/d/fedora/libvirt/f19
#srpm=libvirt-1.0.5.6-3.fc19.src.rpm

# The build script.
#build_script=/tmp/build-it.sh
# Because virt-builder copies the build script permissions too.
#chmod +x $build_script


image_file=/tmp/$guest_type.img

port=8080

# How much guest memory we need for the build:
export LIBGUESTFS_MEMSIZE=4096

# Run virt-builder.
# Use a long build path to work around RHBZ#757089.
# $d/run $d/builder/virt-builder 

virt-builder \
  $guest_type \
  --size 8G \
  --output $image_file \
  --commands-from-file vm-config/0-init \
  --write "/home/user/configure-vm.conf:port=$port" \
  --commands-from-file vm-config/1-user \
  --commands-from-file vm-config/3-packages \
  --firstboot-command 'poweroff'

#  --install /usr/bin/yum-builddep,/usr/bin/rpmbuild,@buildsys-build,@development-tools 
#  --run-command "yum-builddep -y /home/build/$srpm" 

# Run qemu directly.  Could also use virt-install --import here.
qemu-system-x86_64 \
  -nodefconfig \
  -nodefaults \
  -nographic \
  -machine accel=kvm:tcg \
  -cpu host \
  -m 2048 \
  -smp 4 \
  -net user \
  -serial stdio \
  -drive file=$image_file,format=raw,if=virtio,cache=unsafe

# The build ran OK if this contains the magic string (see $build_script).
#virt-cat -a /tmp/$guest_type.img /root/virt-sysprep-firstboot.log

# Copy out the SRPMs & RPMs.
#rm -rf /tmp/result
#mkdir /tmp/result
#virt-copy-out -a /tmp/$guest_type.img /home/build/RPMS /home/build/SRPMS /tmp/result

# Leave the guest around so you can examine the /home/build dir if you want.
# Or you could delete it.
#rm /tmp/$guest_type.img

