#!/bin/bash

. /opt/spack/share/spack/setup-env.sh
source /mnt/deployment/spack_load_setup.sh

spack load postgresql@12.4%gcc@8.3.1 arch=$SPACK_ARCH

for d in $@
do
    echo "[MESSAGE] Doing job in $d, current time: $(date)"
    cd $d
        sed -i 's/\[/\{/g' coincidence_sample.csv
        sed -i 's/\]/\}/g' coincidence_sample.csv
            
        psql -h database -U postgres -d monolithic_crystal -c "\copy gamma_hits from './gamma_hits.csv' delimiter ',' CSV HEADER ;"
        psql -h database -U postgres -d monolithic_crystal -c "\copy coincidence_sample from './coincidence_sample.csv' delimiter ',' CSV HEADER ;"
        psql -h database -U postgres -d monolithic_crystal -c "\copy experiment_coincidence_event from './experiment_coincidence_event.csv' delimiter ',' CSV HEADER ;"
    cd ..
done
