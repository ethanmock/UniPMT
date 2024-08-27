## To get the data

Since the datasets are large, you can download the full datasets from the Google Drive [link](https://drive.google.com/drive/folders/17ht6OBhv34LrZBm9Y2ow_A3KkuAk8tmG?usp=drive_link).

Put the downloaded datasets inside this folder, e.g., `./pmt_pmt/` for the P-M-T dataset.

## Data format

#### NOTE: 
1. task = {pm, pt, pmt}, representing the three tasks in our paper.
2. stage = {train, test}, representing the running stage. The training dataset is only used for training the model, and the test dataset is only used for evaluating the model.

#### Files:
1. files starting with "edges_": contain the main data for the corresponding task, e.g., for pm tasks, the file, `edges_pm_train.csv`, is the training dataset for pm task containing three columns: Peptide, MHC, Label. Note that if there is only one file for the task without "_train/test" as the suffix, the train/test random split is done in the data preprocessing procedure.
2. files starting with "nodes_": contain the mapping between the ID in files with "edges_" with their human-understandable representations. For example, TCR t2: CASGGGGFQETQYF, where t2 is the ID used in the edges file, and CASGGGGFQETQYF is the TCR sequence.
3. `peptides.csv`, `pseudoseqs.csv`, and `tcr.csv`: the initial embedding of peptides, pseudo sequences, and the TCRs discussed in Section 2.1 of our paper.
4. For easy getting the full human-understandable representations of train and test sets, we put the raw datasets in the `raw/` folder

 
#### "edges_" File Example

We take the P-M-T dataset as an example, where the first 10 data samples in "edges_pm_train.csv" in the pmt_pmt dataset are:

| Peptide | MHC | TCR  |
|---------|-----|------|
| p34     | m1  | t367 |
| p3      | m1  | t5310|
| p2      | m1  | t25  |
| p3      | m1  | t5311|
| p3      | m1  | t5312|
| p3      | m1  | t3570|
| p3      | m1  | t3765|
| p3      | m1  | t5313|
| p3      | m1  | t5314|
| p3      | m1  | t5315|

The column "Peptide", "MHC", and "TCR" are represented by their IDs in nodes_peptides.csv, nodes_mhc.csv, and nodes_tcr.csv, respectively. We can obtain their human-understandable representations through the mapping process as follows:

| Peptide    | MHC          | TCR              |
|------------|--------------|------------------|
| GILGFVFTL  | HLA-A*02:01  | CASSSRSSYEQYF     |
| NLVPMVATV  | HLA-A*02:01  | CASSPVTGGIYGYTF   |
| GLCTLVAML  | HLA-A*02:01  | CSARDGTGNGYTF     |
| NLVPMVATV  | HLA-A*02:01  | CASRPDGRETQYF     |
| NLVPMVATV  | HLA-A*02:01  | CASSETGFGNQPQHF   |
| NLVPMVATV  | HLA-A*02:01  | CASSLAPGATNEKLFF  |
| NLVPMVATV  | HLA-A*02:01  | CASSLAPGTTNEKLFF  |
| NLVPMVATV  | HLA-A*02:01  | CASSLGMFNTEAFF    |
| NLVPMVATV  | HLA-A*02:01  | CASSNLPGTVEAFF    |
| NLVPMVATV  | HLA-A*02:01  | CASSPRQSNQPQHF    |

To obtain the full human-understandable representations, please check the `raw/` folder in the pmt_pmt dataset (same for pt_zero and pm_iedb datasets). 
