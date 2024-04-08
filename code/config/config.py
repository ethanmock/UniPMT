import os
import sys
import importlib

"""
Change according to different datasets
==========================================================================
data_folder: the folder of dataset, must be in format ***task_dataname***
files in folder: only the specific task have train/test splitted file, 
others should be ony a whole file to be splitted by code.
==========================================================================
"""

data_folder = 'pt_zeroshot1v1neg'
# whether run test

ablate_pm = False 
ablate_pt = False 
ablate_pmt = False 
 

"""
==============================================================
Following part are some basic config, only use when finetune, 
which are not recommended to change when using the code.
==============================================================
"""
regenerate_graphdata = False
specific_config = True

# combine ratio config, change if needed
pm_weight = 0.1
pt_weight = 0.1
pmt_weight = 0.1

# model config
seed = 3406
batch_size = 512 
epoch_num = 1000
gnn_out_emb_size = 256
emb_size_hid = 1280
hidden_size = 1024
out_emb_size = 256
test_size = 0.2

# basic information of data and model store
dataset = data_folder.split('/')[-1]
task, dataname = dataset.split('_')    # must be task_dataname format 

add_redundance = True 

# data negative sampling config, 0 means no negative sampling
pmt_neg_sample = 0
pm_neg_sample = 0
pt_neg_sample = 0

model_path = f"../output/model/model_{task}_{dataname}.pt"  # must be speccified
directory = os.path.dirname(model_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# evaluate config
pt_threshold = 0.5
pmt_threshold = 0.5
pm_threshold = 0.5


"""
==============================================================
Load data specific config, if False, use default config 
==============================================================
"""
def load_dataset_config(dataset_name):
    """
    Load dataset specific config from f'{dataset_name}.py' to overwrite default config
    """
    try:
        dataset_config_module = importlib.import_module(dataset_name)

        current_module = sys.modules[__name__]

        for key in dir(dataset_config_module):
            if not key.startswith("_"):
                setattr(current_module, key, getattr(dataset_config_module, key))
                print(f"Overwrite {key} to {getattr(dataset_config_module, key)}")

    except ModuleNotFoundError:
        print(f"No specific config found for {dataset_name}, using default config.")

if specific_config:
    # load config from f'{dataname}.py' to overwrite default config
    load_dataset_config(f"config.{dataname}")
    
