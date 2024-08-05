# UniPMT
The code and datasets for the model UniPMT proposed in the paper: UniPMT: A Unified Deep Framework for Peptide, MHC, and TCR Binding Prediction

## To run the code:

### Requirements:

**Operating System**: Linux Ubuntu 20.04.

**Software dependencies and software versions**: please see `./code/requirements.txt`

**Hardware**: CPU: Intel@ Xeon(R) Platinum 8360Y CPU @ 2.40GHzx 144, GPU: Nvidia A100 (for training); Nvidia A100 or Nvidia 3090 (for evaluation)



### Instructions
1. Install the required packages in requirement.py: `pip install -r requirements.txt`. **Normal install time**: within 1 hour.
2. Download the datasets (see in `./data/` folder)，and put the datasets in that folder, e.g., `./data/pmt_pmt/`.
3. Download the model file (see in `./output/model/` folder) and put the trained model file in that folder, e.g., `./output/model/model_pmt_pmt.pt`
 - To reproduce the **PMT** results:
   - Modifiy the `code/config/config.py` file: `data_folder = pmt_pmt`.
   - Run the evaluation through `python main.py`. Expected runing time: within 1 min on Nvidia 3090/5 seconds on Nvidia A100.
- To reproduce the **PM** results:
  - Modifiy the `code/config/config.py`y file: `data_folder = pm_iedbsame`
  - Run the evaluation through `python main.py`. Expected runing time: within 1 min on Nvidia 3090/5 seconds on Nvidia A100.


### Expected output
1. The test results (AUC, PRAUC) in the test set. The results will be outputed on the Terminal.
2. A predicted results scores of each data sample in the test set will be stored in `./output/predictions/`




## Code functinality description (Pseudocode)

**UniPMT Training Process**

1. **Data Processing and Graph Construction**
   - Load and preprocess datasets for P-M, P-T, and P-M-T bindings.
   - Remove duplicates and anomalies from the data.
   - Create edge sets E for P-M, P-T, and P-M-T bindings.
   - Represent peptides (P), MHCs (M), and TCRs (T) as nodes, forming a heterogeneous graph G(V, E).
  
2. **Initial Embedding Representation**
   - Generate initial embeddings for P and T nodes using the ESM method:
     hp, ht <- ESM(P, T)
   - Generate initial embeddings for M nodes using pseudo sequences:
     hm <- Pseudo(M)

3. **Graph Neural Network Learning**
   - **def** _GraphSAGE_:
     - For each node ni at layer l+1:
       h_ni^(l+1) = ReLU(W^(l) * MEAN({h_nj^(l) | nj in Neighbors(ni)}))

4. **Multi-task Learning**
   - **def** _P-M Task Learning:_
     - Generate vector representation for P-M binding:
       v_pm = f_pm(hp, hm)
     - Calculate P-M binding probability:
       P_pm = sigmoid(w_pm * v_pm)
     - Compute cross-entropy loss:
       L_pm = -(1/N_pm) * sum(y_pm^(i) * log(P_pm^(i)) + (1 - y_pm^(i)) * log(1 - P_pm^(i)))

   - **def** _P-M-T Task Learning:_
     - Reuse P-M representation v_pm.
     - Generate vector representation for M-T binding:
       v_mt = f_mt(hm, ht)
     - Calculate P-M-T binding score and probability:
       P_pmt = sigmoid(f_DMF(v_pm * v_mt))
     - Optimize using Info-NCE contrastive learning loss:
       L_pmt = -(1/N_pmt) * sum(log(exp(P_pmt^(i) / tau) / (exp(P_pmt^(i) / tau) + sum(exp(P_pmt^(i,j) / tau))))

   - **def** _P-T Task Learning:_
     - Aggregate P-M binding probabilities:
       P_pt = (1/M) * sum(P_pmjt for j in 1 to M)
     - Compute cross-entropy loss:
       L_pt = -(1/N_pt) * sum(y_pt^(i) * log(P_pt^(i)) + (1 - y_pt^(i)) * log(1 - P_pt^(i)))

5. **Training Process**
   - For each epoch:
     - For each batch in the dataset:
       - Update node embeddings using _GraphSAGE_.
       - Perform _P-M task learning_ and compute L_pm.
       - Perform _P-M-T task learning_ and compute L_pmt.
       - Perform _P-T task learning_ and compute L_pt.
       - L = lambda_pm * L_pm + lambda_pmt * L_pmt + lambda_pt * L_pt
       - Update model parameters through minimizing L.
     - Check for convergence or stopping criteria.
   - Continue training until the model converges or meets predefined stopping criteria.
