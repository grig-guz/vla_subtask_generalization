import copy
from typing import Iterator, Tuple, Any

import glob
from pathlib import Path
from typing import Dict, Tuple, Union
from collections import defaultdict
from tqdm import tqdm
import os
import sys
sys.path.append("../..")
sys.path.append(".")
import numpy as np

import re

import numpy as np


for data_section in ['training']:
    ann_file = np.load(f'/scratch-ssd/gguz/task_D_D/{data_section}/ann_rough_filter/auto_lang_ann.npy', allow_pickle=True)
    new_ann_file = copy.deepcopy(ann_file)
    print(ann_file[()].keys())
    print(ann_file[()]['info'].keys())
    #print(ann_file[()]['info']['indx'])
    #print(ann_file[()]['info']['episodes'])

    print(ann_file[()]['language'].keys())
    #print(ann_file[()]['language']['ann'])
    #print(ann_file[()]['language']['task'])
    #print(ann_file[()]['language']['emb'])

    indices_per_task = defaultdict(list)

    for i, task in enumerate(ann_file[()]['language']['task']):
        print(f"Task: {task}, subtasks: {ann_file[()]['language']['subtasks'][i]}")