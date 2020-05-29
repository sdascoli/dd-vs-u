import torch
import numpy as np
import submitit
from pathlib import Path
from main import main
import collections
import itertools
import time
from argparse import Namespace
import os

folder = 'r.{}'.format(int(time.time()))
if not os.path.exists(folder):
    os.mkdir(folder)

ex = submitit.AutoExecutor(folder)
if ex.cluster == "slurm":
    print("Executor will schedule jobs on Slurm.")
else:
    print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")

widths = np.unique(np.logspace(0, 2.5, 15).astype(int))
ns = np.logspace(1,5,15).astype(int)
grid = collections.OrderedDict({
        'width' : widths,
        'n': ns,
        'depth': [2],
        'wd' : [0., 0.1],
        'activation' : ['tanh'],
        'dataset' : ['random'],
        'noise' : [0,0.5,5],
        'lr' : [0.001],
        'mom' : [0.],
        'd' : [14*14],
        'num_seeds' : [10],
        'test_noise' : [False],
        'loss_type' : ['mse'],
        'epochs' : [10000],
    'no_cuda' : [False],
    'teacher_depth' : [2],
    'teacher_width' : [100],
    'n_test' : [10000],
    'test_noise' : [False],
    'bs' : [1000000],
    'n_classes' : [1],
    'task' : ['regression'],
    'loss_type' : ['mse'],
    })

def dict_product(d):
    keys = d.keys()
    for element in itertools.product(*d.values()):
        yield dict(zip(keys, element))

torch.save(grid, folder + '/params.pkl')

ex.update_parameters(mem_gb=10, nodes=1, tasks_per_node=80, cpus_per_task=1, gpus_per_node=8, timeout_min=1440, slurm_partition='learnfair')
jobs = []
with ex.batch():
    for i, params in enumerate(dict_product(grid)):
        params['name'] = folder+'/{:06d}'.format(i)
        job = ex.submit(main, Namespace(**params))
        jobs.append(job)

