#!/bin/bash

#SBATCH -o %J.out
#SBATCH -e %J.err
#SBATCH -p CPU

. /opt/spack/share/spack/setup-env.sh

if lscpu | grep -q "avx512"; then
   spack load gate@9.0%gcc@8.3.1 arch=linux-centos8-skylake_avx512
   spack load root@6.22.00%gcc@8.3.1 arch=linux-centos8-skylake_avx512
   spack load anaconda3@2019.10%gcc@8.3.1 arch=linux-centos8-skylake_avx512
else
   spack load gate@9.0%gcc@8.3.1 arch=linux-centos8-haswell
   spack load root@6.22.00%gcc@8.3.1 arch=linux-centos8-haswell
   spack load anaconda3@2019.10%gcc@8.3.1 arch=linux-centos8-haswell
fi

source activate monolithic-crystal

root -b -q -l root2csv4.C
python replace_hits_eventID_wit_uuid.py hits_80crystal.csv
