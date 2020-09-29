#!/bin/bash

CPU_MODEL=$(lscpu | grep "Model name:" | awk -F: '{print $2}' | awk '{$1=$1};1') 
echo -e "Working on:\t$(hostname)\nUser:\t$(whoami)\nCPU Model:\t$CPU_MODEL" | column -t -s $'\t'

declare -A cpu_arch_map
############# E5 V4 #############
cpu_arch_map["Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz"]="linux-centos8-broadwell"
cpu_arch_map["Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60GHz"]="linux-centos8-broadwell"

############# E5 V3 #############
cpu_arch_map["Intel(R) Xeon(R) CPU E5-2695 v3 @ 2.30GHz"]="linux-centos8-haswell"

############# Gold  #############
cpu_arch_map["Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz"]="linux-centos8-skylake_avx512"
cpu_arch_map["Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz"]="linux-centos8-skylake_avx512"

############# Core  #############
cpu_arch_map["Intel(R) Core(TM) i7-4930K CPU @ 3.40GHz"]="linux-centos8-ivybridge"

echo "${cpu_arch_map["Intel(R) Core(TM) i7-4930K CPU @ 3.40GHz"]}"
for key in ${!cpu_arch_map[@]}; do echo $key; done
for value in ${cpu_arch_map[@]}; do echo $value; done
echo cpu_arch_map has ${#cpu_arch_map[@]} elements

