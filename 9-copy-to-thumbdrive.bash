#! /usr/bin/bash
target=$1

if [ -z "$target" ]; then
  echo "Need to specify a target path"
  exit 1
fi

echo "Target: '${target}/'"
#exit 1

most_recent_ova=`ls ./vm-images/*.ova | tail -1`
if [ -z "$most_recent_ova" ]; then
  echo "Need to have created a .ova file first"
  exit 1
fi

# target ~ /run/media/andrewsm/591F-4AF0

rsync -avz --progress ${most_recent_ova} ${target}/

# Windows and OSX binaries
rsync -avz --progress ./vm-images/VirtualBox-5* ${target}/

# Linux binaries (Fedora, and some Ubuntus)
rsync -avz --progress ./vm-images/virtualbox-5* ${target}/

ls -l ${target}/
