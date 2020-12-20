#!/bin/bash

#. /opt/spack/share/spack/setup-env.sh
#source /mnt/deployment/spack_load_setup.sh

#spack load root@6.22.00%gcc@7.3.0 arch=$SPACK_ARCH
#spack load anaconda3@2019.10%gcc@8.3.1 arch=$SPACK_ARCH
#source activate monolithic-crystal

for d in $@
do
    #echo "[MESSAGE] Doing job in $d, current time: $(date)"
    cd $d
    #time root -b -q -l root2csv4.C
    #time python optical_to_sample.py hits.csv <experiment_id>
    #sbatch -w s7 sbatch_task.sh
    sbatch --mem=10240 sbatch_task.sh
    
    cd ..
done
