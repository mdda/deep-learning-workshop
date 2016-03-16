#!/bin/bash -

# Pop up a password question box, and write the result to :
# ./config-vm-host/root-password

DIALOGTEXT="Please type in a root password for the VM"

echo `zenity --title 'EncFS Password' --entry --hide-text --text "$DIALOGTEXT"` > ./config-vm-host/root-password
chmod 600 ./config-vm-host/root-password
