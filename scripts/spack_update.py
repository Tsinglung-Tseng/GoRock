from pifs.cli_command import SpackFind
import os


os.environ["PICLUSTER_DB"] = "postgresql://picluster@192.168.1.96:5432/picluster"

SpackFind().commit()
