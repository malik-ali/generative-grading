import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from shutil import copyfile

import getpass
USER = getpass.getuser()

NUM_LABELS = 13
ROOT = f'/mnt/fs5/{USER}/generative-grading/pyramidSnapshot_v2'


def create_folders(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        for i in range(NUM_LABELS):
            os.makedirs(path + "/samp" + str(i + 1))
    return path


def main():
    for i in range(13):
        ret = []
        allsamps = []
        for image_path in glob.glob(os.path.join(ROOT, 'all', 'samp'+ str(i+1) +'/*.png')):
            allsamps.append(image_path)
        print()
        random.shuffle(allsamps)
        val_len = int(len(allsamps)*0.2)

        val = allsamps[0:val_len]
        train = allsamps[val_len:]

        for j in tqdm(range(len(val))):
            image_path = val[j]
            save_name = os.path.basename(os.path.normpath(image_path))
            dir_name = os.path.join(ROOT, 'cross_val', 'samp'+str(i+1))
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            copyfile(image_path, os.path.join(dir_name, save_name))


        for k in tqdm(range(len(train))):
            image_path = train[k]
            save_name = os.path.basename(os.path.normpath(image_path))
            dir_name = os.path.join(ROOT, 'train', 'samp'+str(i+1))
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            copyfile(image_path, os.path.join(dir_name, save_name))


if __name__ == '__main__':
    main()

