#!/bin/bash

. /opt/spack/share/spack/setup-env.sh
source /mnt/deployment/spack_load_setup.sh

spack load postgresql arch=$SPACK_ARCH
spack load root@6.22.00%gcc@7.3.0 arch=$SPACK_ARCH
spack load anaconda3@2019.10%gcc@8.3.1 arch=$SPACK_ARCH
source activate monolithic-crystal

for d in $@
do
    echo "[MESSAGE] Doing job in $d, current time: $(date)"
    #cp /home/zengqinglong/scaffold/GoRock/scripts/data_process/{root2csv4.C,replace_hits_eventID_wit_uuid.py,optical_to_sample.py} $d
    cd $d
    time root -b -q -l root2csv4.C
    time python replace_hits_eventID_wit_uuid.py hits.csv
    time python optical_to_sample.py hits.csv
    cd ..
done
