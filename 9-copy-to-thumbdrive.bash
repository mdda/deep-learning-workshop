#! /usr/bin/bash
target=$1

if [ -z "$target" ]; then
  echo "Need to specify a target path"
  exit 1
fi

echo "Target: '${target}/'"

## As user :
# id -a

## As root :
# mkdir /media/usbdrive
# mount -o uid=1000,gid=1000 /dev/sdb1 /media/usbdrive/

## As user :
# ./9-copy-to-thumbdrive.bash /media/usbdrive/

## Finally, as root :
# umount /media/usbdrive/

## and repeat the cycle for each key required...

most_recent_ova=`ls ./vm-images/*.ova | tail -1`
if [ -z "$most_recent_ova" ]; then
  echo "Need to have created a .ova file first"
  exit 1
fi
#exit 1

# target ~ /run/media/andrewsm/591F-4AF0

rm -f ${target}/*.txt
INSTRUCTIONS=${target}/0_INSTRUCTIONS.txt

cat >${INSTRUCTIONS} <<'EOT'

Deep-Learning-Workshop "Hands-on" Materials
-------------------------------------------

Please do each of the following steps :

1) Create a new folder on your laptop's hard drive

2) Copy the /presentation folder on the USB drive into your new folder

3) Copy the '.ova' file  on the USB drive into your new folder

4) If you *don't* have VirtualBox installed : 
    a) You should have read the workshop description more carefully; 
    b) You might be able to use an installation file from the /virtualbox-install/ folder; and
    c) Good luck...

5) Launch a browser on your copy of /presentation/index.html

6) Pass the key to someone else who needs it - or back to the speaker...

EOT


VBOXINSTALL=${target}/virtualbox-install
mkdir -p ${VBOXINSTALL}

# Windows and OSX binaries
rsync -avz --progress ./vm-images/VirtualBox-5* ${VBOXINSTALL}/

# Linux binaries (Fedora, and some Ubuntus)
rsync -avz --progress ./vm-images/virtualbox-5* ${VBOXINSTALL}/

# Now the presentation materials
rsync -avz --progress ./presentation ${target}/

# Clean out old VMs
#rm ${target}/deep-learning-workshop_2016-06-23*.ova
#rm ${target}/deep-learning-workshop_2016-07-21*.ova
rm ${target}/deep-learning-workshop_2016-*.ova

# And ensure the new one is there
rsync -avz --progress ${most_recent_ova} ${target}/
ls -l ${target}/

#echo "5da875cf8e0df504e5edcccf3382630f  vm-images/deep-learning-workshop_2016-07-21_18-06.ova"
#echo "6f5d8259872b6cae9d2b23be8012fc4d  vm-images/deep-learning-workshop_2016-07-28_11-40.ova"
#echo "2605356947c87cd1052e782ebe163016  vm-images/deep-learning-workshop_2017-01-23_00-12.ova"
echo "e81531f2d3fcaa99fb74a21199e9974c  /media/usbdrive//deep-learning-workshop_2017-03-03_01-09.ova"

md5sum ${target}/*.ova 
