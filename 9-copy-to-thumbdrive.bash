#! /usr/bin/bash
target=$1

if [ -z "$target" ]; then
  echo "Need to specify a target path"
  exit 1
fi

echo "Target: '${target}/'"

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
rm ${target}/deep-learning-workshop_2016-06-23*.ova

# And ensure the new one is there
rsync -avz --progress ${most_recent_ova} ${target}/
ls -l ${target}/

