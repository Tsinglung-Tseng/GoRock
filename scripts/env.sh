#!/bin/bash

export DB_CONNECTION="postgresql://postgres:postgres@192.168.1.185:54322/incident"
export DEFAULT_MODEL_ROOT="/clusterfs/users/qinglong/.runs/models_common/"

conda activate incident-estimation