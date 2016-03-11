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

qemu-img convert -O vmdk ${image_file} ${vbox_disk}

# ls -l vm-images/
## -rw-r--r--. 1 andrewsm andrewsm 6442450944 Mar 11 22:05 fedora-23.img
## -rw-r--r--. 1 andrewsm andrewsm 1944453120 Mar 11 23:07 fedora-23_fossasia.vmdk

## Next : Create a linux-fedora-x64 machine with that vmdk disk image
##        Set up port forwarding on the network adapter
##        Export virtual machine appliance : OVA

## The OVA created (which contains the vmdk, which contains the .img ...) :
#ls -l vm-images/
#-rw-r--r--. 1 andrewsm andrewsm 6442450944 Mar 11 22:05 fedora-23.img
#-rw-r--r--. 1 andrewsm andrewsm 1945305088 Mar 11 23:13 fedora-23_fossasia.vmdk
#-rw-------. 1 andrewsm andrewsm  817735168 Mar 11 23:15 fedora-23_fossasia.ova


## After importing the OVA :
# [andrewsm@square fossasia-2016_deep-learning]$ ls -l  /home/andrewsm/VirtualBoxVMs/fossasia_1/
# total 1904036
# -rw-------. 1 andrewsm andrewsm       7918 Mar 11 23:22 fossasia_1.vbox
# -rw-------. 1 andrewsm andrewsm       7918 Mar 11 23:20 fossasia_1.vbox-prev
# -rw-------. 1 andrewsm andrewsm 1950220288 Mar 11 23:22 fossasia-disk1.vmdk

