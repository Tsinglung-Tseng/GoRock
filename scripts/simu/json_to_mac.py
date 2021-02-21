import os
import argparse
from hotpot.database import Database
from hotpot.simulation.mac import MAC


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ[
    "DB_CONNECTION"
] = "postgresql://postgres@192.168.1.96:5432/monolithic_crystal"
os.environ["PICLUSTER_DB"] = "postgresql://picluster@192.168.1.96:5432/picluster"

selected_config = MAC.from_database(1)
print(selected_config.dump())
