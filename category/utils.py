import numpy as np
import os
import os.path as osp
import argparse

Config ={}
Config['root_path'] = '/home/ec2-user/workspace/ee599-hw4/data/polyvore_outfits/'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''


Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 50
Config['batch_size'] = 128

Config['learning_rate'] = 1e-3
Config['num_workers'] = 8