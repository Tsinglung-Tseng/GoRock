import argparse
import subprocess
from hotpot.database import Database
from hotpot.simulation import MAC


parser = argparse.ArgumentParser(description="Quick make new simu for more data.")
parser.add_argument(
    "geometry_id", type=int, nargs=1, help="Geometry id of this simulation."
)

parser.add_argument(
    "source_id", type=int, nargs=1, help="Source id of this simulation."
)

parser.add_argument(
    "--sum",
    dest="accumulate",
    action="store_const",
    const=sum,
    default=max,
    help="sum the integers (default: find the max)",
)

args = parser.parse_args()
geometry_id = args.geometry_id[0]
source_id = args.source_id[0]

print(
    f"[MESSAGE] Creating new simu using source_id: {source_id}, geometry_id: {geometry_id}"
)


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ[
    "DB_CONNECTION"
] = "postgresql://postgres@192.168.1.96:5432/monolithic_crystal"
os.environ["PICLUSTER_DB"] = "postgresql://picluster@192.168.1.96:5432/picluster"


work_config_template = {
    "work_dir": "/home/zengqinglong/optical_simu/dev_simu_work_dir",
    "commit_message": "built by simu creater, developing.",
    "number_of_subs": 10,
    "work_dir_parser": sub_work_dir_selector_by_name_pattern,
}


def sub_work_dir_selector_by_name_pattern(pattern="sub.*"):

    return subprocess.call()

def parpare_work_env(work_dir, commit_message, number_of_subs, work_dir_parser):
    def copy_resource():
        return

    def make_work_space():
        return

    make_work_space(work_dir, commit_message, number_of_subs)
    copy_resource(simu_resource_collector(geometry_id, source_id), work_dir_parser)
    return


def work_software_env():

    return


def run_work
