# UniPMT
The code and datasets for the model UniPMT proposed in the paper: UniPMT: A Unified Deep Framework for Peptide, MHC, and TCR Binding Prediction

## To run the code:

### Requirements:

**Operating System**: Linux Ubuntu 20.04.

**Software dependencies and software versions**: please see `./code/requirements.txt`

**Hardware**: CPU: Intel@ Xeon(R) Platinum 8360Y CPU @ 2.40GHzx 144, GPU: Nvidia A100



### Instructions
1. Install the required packages in requirement.py: `pip install -r requirements.txt`. **Normal install time**: within 1 hour.
2. Download the datasets (see in `./data/` folder)，and put the datasets in that folder, e.g., `./data/pmt_pmt/`.
3. Specify the config file (`code/config/config.py`):
   - a Set the dataset to run, e.g., `data_folder = pmt_pmt`.
   - b Set whether to run ablate study. e.g., `ablate_pm = True` means no using PM information. All set to `False` means using the full UniPMT.
   - c Dataset specific configs are in their corresponding config files, e.g., `./config/pmt.py`, which are not recommended to modify.
4. Run the evaluation through `python main.py`. **Expected run time**: within 1 min.


### Expected output
1. The test results (AUC, PRAUC) in the test set. The results will be outputed on the Terminal.
2. A predicted results scores of each data sample in the test set will be stored in `./output/predictions/`



