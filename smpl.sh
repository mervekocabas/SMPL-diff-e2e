#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


# # SMPL body model for evaluation of 3DPW/RICH/H3.6M
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/body_models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O './data/body_models/smpl.zip' --no-check-certificate --continue
unzip data/body_models/smpl.zip -d data/body_models/
mv data/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_FEMALE.pkl
mv data/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_MALE.pkl
mv data/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_NEUTRAL.pkl
