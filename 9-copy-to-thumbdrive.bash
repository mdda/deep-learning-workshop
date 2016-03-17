#! /usr/bin/bash
target=$1

if [ "$target" -e "" ]; then
  echo "Need to specify a target path"
  exit(1)
fi

# target ~ /run/media/andrewsm/591F-4AF0

rsync -avz --progress ./vm-images/fossasia-mdda_*.ova target/
rsync -avz --progress ./vm-images/VirtualBox-5* target/
rsync -avz --progress ./vm-images/virtualbox-5* target/
