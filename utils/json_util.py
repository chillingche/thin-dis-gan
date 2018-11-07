import os
import json


def delete(json_path):
    if os.path.exists(json_path):
        os.remove(json_path)


def dump(obj, json_path):
    if os.path.exists(json_path):
        obj = load(json_path) + obj
    with open(json_path, 'w') as f:
        json.dump(obj, f)


def load(json_path):
    with open(json_path, 'r') as f:
        obj = json.load(f)
    return obj
