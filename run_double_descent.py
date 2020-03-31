import os
import time
import subprocess
import itertools
import argparse
import torch
import numpy as np

from utils import copy_py, who_am_i

def create_script(params):
    script = '''#!/bin/bash 
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=run.out
#SBATCH --job-name=run

python main.py --noise {name} --n {n}
'''.format(**params)
    with open('run.sbatch', 'w') as f:
        f.write(script)

def send_script(file):
    process = subprocess.Popen(['sbatch', file], stdout=subprocess.PIPE)

if __name__ == '__main__':
    
    exp_dir = 'r.{}'.format(int(time.time()))
    os.mkdir(exp_dir)
    copy_py(exp_dir) 
    os.chdir(exp_dir)

    grid = {
            'noise' : np.logspace(-3,-1,10),
            'lr' : np.logspace(-2,0,3).astype(int),
            'n' : np.logspace(1,3,10).astype(int),
            'd' : [1,10,100],
            'seed' : range(5)
    }

    def dict_product(d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))

    for params in dict_product(grid):
        # one can read all .params files in a folder to get the names
        params['name'] = '{noise:.2f}_{n:.2f}_{d:.2f}_{}'.format(**params)
        create_script(params)
        file_name = '{}.sbatch'.format(params['name'])
        send_script(file_name)


