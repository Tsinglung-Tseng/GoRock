import os
from typing import Iterable
from functools import reduce
import operator
from .database import Database
from .simulation.mac import MAC


class CMD:
    """
    A command line utility. Provide naivity ways to pipelining commands.

    Preview command built with CMD:
    >>> CMD.sub_finder(task_config).pipe(
    >>>     CMD.sort_numerically()
    >>> )

    >>> <CMD> -- command body: find /home/zengqinglong/optical_simu/dev_simu_work_dir -type d -name sub.* | sort -V

    Run a CMD command:
    >>> CMD.check_task_output(task_config).pipe(
    >>>     CMD.sort_numerically()
    >>> ).run()
    """
    def __init__(self, cmd_body):
        self.cmd_body = cmd_body
    
    def __repr__(self):
        return f"""<{self.__class__.__name__}> -- command body: {self.cmd_body}"""
        
    def pipe(self, next_cmd):
        return CMD(" | ".join([self.cmd_body, next_cmd.cmd_body]))
    
    def atom(self, next_cmd):
        return CMD(" && ".join([self.cmd_body, next_cmd.cmd_body]))
    
    def run(self):
        return os.popen(self.cmd_body).read().split('\n')[:-1]
    
    @staticmethod
    def sub_finder(task_config):
        return CMD(f'''find {task_config["work_dir"]} -type d -name {task_config["sub_pattern"]}''')

    @staticmethod
    def sort_numerically():
        return CMD("sort -V")
    
    @staticmethod
    def check_task_output(task_config):
        return (
            CMD(f"""find {task_config["work_dir"]} -type d -name "{task_config["sub_pattern"]}" \! -exec test -e '""" 
                + "{}" 
                + f"""/{task_config["task_output"]}' \; -print""")
        )
    
    @staticmethod
    def make_work_dir(task_config):
        return CMD(f"mkdir {task_config['work_dir']}").atom(
            CMD(f"cd {task_config['work_dir']}")
        ).atom(
            CMD(f"mkdir task_{task_config['number_of_subs']}_subs")
        ).atom(
            CMD(f"cd task_{task_config['number_of_subs']}_subs")
        ).atom(
            CMD(f"seq 0 {task_config['number_of_subs']-1} | xargs -i mkdir sub."+"{}")
        )


def prepare_script(task_config):
    with open("/tmp/sbatch_task.sh", 'w') as f:
        f.write(task_config["task_script"](task_config["task_id"]))

def prepare_mac(task_config):
    source_path = Database().read_sql(f'''select path from experiment_config where experiment_id={task_config['geometry_id']};''').to_numpy()[0][0]+'/sub.0/'
    with open("/tmp/Geometry.mac", 'w') as f:
        f.write(MAC.from_file(source_path+'Geometry.mac').raw_mac)
        
    with open("/tmp/Source.mac", 'w') as f:
        f.write(MAC.from_file(source_path+'Source.mac').raw_mac)
        
def load_source(task_config):
    return CMD(f'''find {task_config['work_dir']}/task_{task_config['number_of_subs']}_subs -type d -name "sub.*" | xargs -i cp /tmp/''' 
        + '{sbatch_task.sh,Geometry.mac,Source.mac}' 
        + ' {}').atom(
        CMD(f'''find {task_config['work_dir']}/task_{task_config['number_of_subs']}_subs -type d -name "sub.*" | xargs -i cp /home/zengqinglong/optical_simu/optical_common/*''' 
        + ' {}')
    )

def submit_to_slurm(task_config):
    return CMD.check_task_output(task_config).pipe(
        CMD.sort_numerically()
    ).pipe(
        CMD("""xargs -L 1 bash -c 'cd "$0" && pwd && sbatch --cpus-per-task=8 --mem 20240 sbatch_task.sh'""")
    )
