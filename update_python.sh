#!/bin/sh

wget https://raw.githubusercontent.com/wonyl11/FW2022CS655-GENI_mini_project/main/code/Client.py
wget https://raw.githubusercontent.com/wonyl11/FW2022CS655-GENI_mini_project/main/code/client_params

wget https://raw.githubusercontent.com/wonyl11/FW2022CS655-GENI_mini_project/main/code/Institution.py
wget https://raw.githubusercontent.com/wonyl11/FW2022CS655-GENI_mini_project/main/code/institution_params

wget https://raw.githubusercontent.com/wonyl11/FW2022CS655-GENI_mini_project/main/code/ParameterServer.py
wget https://raw.githubusercontent.com/wonyl11/FW2022CS655-GENI_mini_project/main/code/paramsvr_params

wget https://raw.githubusercontent.com/wonyl11/FW2022CS655-GENI_mini_project/main/requirements.txt

add-apt-repository ppa:deadsnakes/ppa
apt update
apt-get install python3.10
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2

sudo apt-get remove --purge python-pip
apt install python3-pip
pip3 install -r requirements.txt
