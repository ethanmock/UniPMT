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
3. Download the model file (see in `./output/model/` folder) and put the trained model file in that folder, e.g., `./output/model/model_pmt_pmt.pt`
 - To reproduce the **PMT** results:
   - Modifiy the `code/config/config.py` file: `data_folder = pmt_pmt`.
   - Run the evaluation through `python main.py`. Expected runing time: within 1 min.
- To reproduce the **PM** results:
  - Modifiy the `code/config/config.py`y file: `data_folder = pm_iedbsame`
  - Run the evaluation through `python main.py`. Expected runing time: within 1 min.


### Expected output
1. The test results (AUC, PRAUC) in the test set. The results will be outputed on the Terminal.
2. A predicted results scores of each data sample in the test set will be stored in `./output/predictions/`



