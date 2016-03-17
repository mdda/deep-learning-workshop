#! /usr/bin/bash
target=$1

if [ -z "$target" ]; then
  echo "Need to specify a target path"
  exit 1
fi

echo "Target: '${target}/'"
#exit 1

# target ~ /run/media/andrewsm/591F-4AF0

rsync -avz --progress ./vm-images/fossasia-mdda_*.ova target/
rsync -avz --progress ./vm-images/VirtualBox-5* target/
rsync -avz --progress ./vm-images/virtualbox-5* target/
