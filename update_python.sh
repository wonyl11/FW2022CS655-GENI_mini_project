#!/bin/sh

add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.10
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
wget https://bootstrap.pypa.io/get-pip.py
sudo apt-get remove --purge python-pip
python3 get-pip.py
pip3 install -r requirements.txt
