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

if [ ! -e "${vbox_disk}" ]; then
  qemu-img convert -O vmdk ${image_file} ${vbox_disk}
fi

# ls -l vm-images/
## -rw-r--r--. 1 andrewsm andrewsm 6442450944 Mar 11 22:05 fedora-23.img
## -rw-r--r--. 1 andrewsm andrewsm 1944453120 Mar 11 23:07 fedora-23_fossasia.vmdk

## Next : Create a linux-fedora-x64 machine with that vmdk disk image
##        Set up port forwarding on the network adapter
##        Export virtual machine appliance : OVA

## The OVA created (which contains the vmdk, which contains the .img ...) :
# ls -l vm-images/
# -rw-r--r--. 1 andrewsm andrewsm 6442450944 Mar 11 22:05 fedora-23.img
# -rw-r--r--. 1 andrewsm andrewsm 1945305088 Mar 11 23:13 fedora-23_fossasia.vmdk
# -rw-------. 1 andrewsm andrewsm  817735168 Mar 11 23:15 fedora-23_fossasia.ova


## After importing the OVA :
# ls -l  /home/andrewsm/VirtualBoxVMs/fossasia_1/
# total 1904036
# -rw-------. 1 andrewsm andrewsm       7918 Mar 11 23:22 fossasia_1.vbox
# -rw-------. 1 andrewsm andrewsm       7918 Mar 11 23:20 fossasia_1.vbox-prev
# -rw-------. 1 andrewsm andrewsm 1950220288 Mar 11 23:22 fossasia-disk1.vmdk


## http://www.linuxhomeserverguide.com/server-config/CreateVM.php
#VBoxManage createvm --name [nameofVM] --register
#VBoxManage modifyvm [nameofVM] --memory 1024 --acpi on --boot1 dvd --nic1 bridged --bridgeadapter1 eth0
#VBoxManage createhd --filename [nameofdisk].vdi --size 10000
#VBoxManage storagectl [nameofVM] --name "Sata Controller" --add sata
#VBoxManage storageattach [nameofVM] --storagectl "Sata Controller" --port 0 --device 0 --type hdd --medium [nameofdisk].vdi
#VBoxManage storageattach [nameofVM] --storagectl "Sata Controller" --port 1 --device 0 --type dvddrive --medium  [/full/path/to/iso/file.iso]
#VBoxManage modifyvm [nameofVM] --vrdeport 3001

## To find --ostype "" potentials: 
# VBoxManage list ostypes

VBoxManage createvm --name ${vbox_name} --ostype ${vbox_ostype} --register

VBoxManage modifyvm ${vbox_name} --memory ${vbox_memory} --acpi on \
       --natpf1     jupyter,tcp,,${port_jupyter},,${port_jupyter} \
       --natpf2 tensroboard,tcp,,${port_tensorboard},,${port_tensorboard}
                                          
#       --nic1 bridged --bridgeadapter1 eth0  
# [--natpf<1-N> [<rulename>],tcp|udp,[<hostip>],
#        <hostport>,[<guestip>],<guestport>]

#--boot1 dvd  --vrdeport 3389

VBoxManage storagectl ${vbox_name} --name "Sata Controller" --add sata
VBoxManage storageattach ${vbox_name} --storagectl "Sata Controller" --port 0 --device 0 --type hdd --medium ${vbox_disk}

VBoxManage export ${vbox_name} --output ${vbox_appliance} --ovf10 \
   --vsys 0 \
   --vendor "Red Cat Labs" --vendorurl "http://www.redcatlabs.com/" \
   --description "Deep Learning Workshop at FOSSASIA 2016 (Singapore)"

## TODO :
#VBoxManage unregistervm ${vbox_name}
