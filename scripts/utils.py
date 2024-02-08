import os
import json


def create_dir(dir): 
    if not os.path.exists(dir): 
        os.makedirs(dir)

def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)  
        f.close()
    return data