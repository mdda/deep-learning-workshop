#!/bin/bash -

# Pop up a password question box, and write the result to :
# ./vm-config/root-password


DIALOGTEXT="Please type in a root password for the VM"

echo `zenity --title 'EncFS Password' --entry --hide-text --text '$DIALOGTEXT'` > ./vm-config/root-password
