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

data_folder = 'pmt_pmt'
# whether run test
run_test = False

ablate_pm = False
ablate_pt = False
ablate_pmt = False

pmt_run_negative = 10

"""
==============================================================
Following part are some basic config, only use when finetune, 
which are not recommended to change when using the code.
==============================================================
"""
regenerate_graphdata = False 
specific_config = True

# sample information
sample_m = False 
sample_num = 1

# combine ratio config, change if needed
pm_weight = 0.1
pt_weight = 0.1
pmt_weight = 0.1

# model config
seed = 3406
batch_size = 256 
epoch_num = 1000
lr = 0.00001
reg_lambda = 0.00001
gnn_out_emb_size = 1280 
emb_size_hid = gnn_out_emb_size 
hidden_size = 1280 
out_emb_size = gnn_out_emb_size 
test_size = 0.2

# basic information of data and model store
dataset = data_folder.split('/')[-1]
task, dataname = dataset.split('_')    # must be task_dataname format 

add_redundance = True 

# data negative sampling config, 0 means no negative sampling
pmt_neg_sample = 0
pm_neg_sample = 0
pt_neg_sample = 0

# train config
## get current time in format: 2019-12-12-12-12-12
#start_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
model_path = f"../model/model_{task}_{dataname}.pt"  # if run_test = True, must be speccified
directory = os.path.dirname(model_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# evaluate config
pt_threshold = 0.5
pmt_threshold = 0.5
pm_threshold = 0.5

# test config
test_model = f"../output/model/model_{task}_{dataname}.pt"  # if run_test = True, must be speccified
test_file_name = "example"
test_file = f"../predict/{test_file_name}.csv"
test_result = f"../predict/result_{test_file_name}.csv"

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
    
