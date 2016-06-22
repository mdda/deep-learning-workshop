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

${VBOXINSTALL}=${target}/virtualbox-install
mkdir -p ${VBOXINSTALL}

# Windows and OSX binaries
rsync -avz --progress ./vm-images/VirtualBox-5* ${VBOXINSTALL}/

# Linux binaries (Fedora, and some Ubuntus)
rsync -avz --progress ./vm-images/virtualbox-5* ${VBOXINSTALL}/

# Now the presentation materials
rsync -avz --progress ./presentation ${target}/

ls -l ${target}/

rm ${target}/*.ova

exit 1


rsync -avz --progress ${most_recent_ova} ${target}/

ls -l ${target}/

