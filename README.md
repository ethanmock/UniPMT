# UniPMT
The code for the model UniPMT proposed in the paper: UniPMT: A Unified Deep Framework for Peptide, MHC, and TCR Binding Prediction

## To run the code:

### Requirements:

**Operating System**: Linux Ubuntu 20.04.

**Software dependencies and software versions**: please see `./code/requirements.txt`

**Hardware**: CPU: Intel@ Xeon(R) Platinum 8360Y CPU @ 2.40GHzx 144, GPU: Nvidia A100



### Instructions
1. Install the required packages in requirement.py: `pip install -r requirements.txt`. **Normal install time**: within 1 hour.
2. Input the dataset to run in "data_folder" variable in code/config/config.py (e.g., `data_folder = pt_zeroshot1v1neg`)
3. Run the evaluation through `python main.py`. **Expected run time**: within 1 min.


### Expected output
1. The test results (AUC, PRAUC) in the test set. The results will be outputed on the Terminal.
2. A predicted results scores of each data sample in the test set will be stored in `./output/predictions/`



