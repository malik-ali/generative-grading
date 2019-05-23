import csv
import pickle
import json
import pathlib
import numpy as np
import tempfile, shutil, os, uuid

import scipy.io as scio

def save_pickle(obj, f_path):
    with open(f_path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(f_path):
    with open(f_path, 'rb') as f:
        return pickle.load(f)

def save_json(obj, f_path):
    with open(f_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)

def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)

def save_np(obj, f_path):
    np.save(f_path, obj)

def load_np(f_path):
    return np.load(f_path)

# Good for saving multiple np arrays in a dict, with names for keys
def savemat(dic, f_path):
    scio.savemat(f_path, dic)

def loadmat(f_path):
    return scio.loadmat(f_path)

def read_csv_dict(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def create_temp_copy(path):
    temp_dir = tempfile.gettempdir()
    path_prefix, suffix = os.path.splitext(path)
    filename = os.path.basename(path_prefix) + '_' + str(uuid.uuid1()) + suffix
    
    temp_path = os.path.join(temp_dir, filename)
    shutil.copy2(path, temp_path)
    return temp_path
