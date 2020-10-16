#!/bin/bash

#SBATCH -o %J.out
#SBATCH -e %J.err
#SBATCH -p CPU

 source /mnt/deployment/spack_load_setup.sh

. /opt/spack/share/spack/setup-env.sh

# spack load gate@8.2 arch=$SPACK_ARCH
# spack load root@6.22.00%gcc@8.3.1 arch=$SPACK_ARCH
spack load anaconda3@2019.10%gcc@8.3.1 arch=$SPACK_ARCH
spack load postgresql@12.4%gcc@8.3.1 arch=$SPACK_ARCH

# if lscpu | grep -q "avx512"; then
#    spack load gate@9.0%gcc@8.3.1 arch=linux-centos8-skylake_avx512
#    spack load root@6.22.00%gcc@8.3.1 arch=linux-centos8-skylake_avx512
#    spack load anaconda3@2019.10%gcc@8.3.1 arch=linux-centos8-skylake_avx512
# else
#    spack load gate@9.0%gcc@8.3.1 arch=linux-centos8-haswell
#    spack load root@6.22.00%gcc@8.3.1 arch=linux-centos8-haswell
#    spack load anaconda3@2019.10%gcc@8.3.1 arch=linux-centos8-haswell
# fi

#source picluster load postgresql
source activate monolithic-crystal

#root -b -q -l root2csv4.C
#python replace_hits_eventID_wit_uuid.py hits_80crystal.csv
python optical_to_sample.py hits_80crystal.csv
sed -i 's/\[/\{/g' sample.csv
sed -i 's/\]/\}/g' sample.csv
psql -h database -U postgres -d monolithic_crystal -c "\copy coincidence_sample from './sample.csv' delimiter ',' CSV HEADER ;"
