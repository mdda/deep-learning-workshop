#!/bin/bash -

set -e
set -x

source ./vm-config/params

## rsync -avz --progress andrewsm@simlim:/home/andrewsm/OpenSource/fossasia-2016_deep-learning/vm-images/*.img ./vm-images/
#receiving incremental file list
#fedora-23.img
#  6,442,450,944 100%   33.28MB/s    0:03:04 (xfr#1, to-chk=0/1)
#sent 43 bytes  received 800,696,292 bytes  4,316,422.29 bytes/sec
#total size is 6,442,450,944  speedup is 8.05
# ->> Apparently the image is pretty empty inside...

## Needs VirtualBox installed on the machine...


# VBoxManage convertfromraw --format vmdk --variant Standard <source>.vmdk <destination>.vdi

# http://it-ovid.blogspot.sg/2012/10/virtual-box-headless-cheatsheet.html

# Want to IMPORT APPLIANCE : *.ovf

## Looks promising : http://virtuallyhyper.com/2013/06/migrate-from-libvirt-kvm-to-virtualbox/
# Except that the virt-convert tool no longer supports the -o vmx option

qemu-img info ${image_file}

## image: ./vm-images/fedora-23.img
## file format: raw
## virtual size: 6.0G (6442450944 bytes)
## disk size: 6.0G

# http://softwarerecs.stackexchange.com/questions/30424/open-source-commandline-tool-to-create-ovf-and-ova-files
#   "Oracle Virtual Box can export to OVF files and VBoxManage clonehd can convert VMDK to streaming VMDK amongst many other options."

qemu-img convert ${image_file} ${vbox_disk}
