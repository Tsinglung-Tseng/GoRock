from pifs.cli_command import SpackFind
os.environ["PICLUSTER_DB"] ="postgresql://picluster@192.168.1.96:5432/picluster"

SpackFind().commit()
