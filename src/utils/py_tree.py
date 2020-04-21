import json
from dotmap import DotMap
from picluster.primitives.FS import File


def is_json_file_url(input):
    try:
        File(url=input) 
        return True
    return False


def json_to_obj(j: "dict | str"):
    if is_json_file_url(j):
        jf = open(azure_config_json)
        jf_str = jf.read()
        return DotMap(json.loads(jf_str))
    
    if is_dict:
        #TODO
        return
