import numpy as np
import os
import os.path as osp
import argparse

Config ={}
Config['root_path'] = '/home/ec2-user/workspace/ee599-hw4/data/polyvore_outfits/'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''


Config['use_cuda'] = True
Config['debug'] = True
Config['num_epochs'] = 2
Config['batch_size'] = 500

Config['learning_rate'] = 0.001
Config['num_workers'] = 32