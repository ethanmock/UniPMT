## To get the data

Since the datasets are large, you can download the full datasets from the google drive [link](https://drive.google.com/drive/folders/17ht6OBhv34LrZBm9Y2ow_A3KkuAk8tmG?usp=drive_link).

Put the downloaded datasets inside this folder, e.g., `./pmt_pmt/`.

## Data format

#### NOTE: 
1. task = {pm, pt, pmt}, representing the three tasks in our paper.
2. stage = {train, test}, representing the running stage, the train dataset is only used for training the model, and the test dataset is only used for evaluating the model.

#### Files:
1. files starting with "edges_": contain the main data for the corresponding task, e.g., for pm tasks, the file, "edges_pm_train.csv", is the training dataset for pm task containing three columns: Peptide, MHC, Label. Note that if there is only one file for the task without "_train/test" as the sufix, the train/test random split is done in data preprocessing procedure.
2. files starting with "nodes_": contain the mapping between the ID in files with "edges_" with their human-understandable representations. For example, TCR t2: CASGGGGFQETQYF, where t2 is the ID used in edges file and CASGGGGFQETQYF is the TCR sequence.
3. "peptides.csv", "pseudoseqs.csv", and "tcr.csv": the initial embedding of peptides, pseudoseqs and tcr discussed in Section 2.1 of our paper.


#### "edges_" File Example

The first 10 data samples in "edges_pm_train.csv" in pmt_pmt dataset are:

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

The column "Peptide", "MHC", "TCR" are represented by their ID. We can obtain their human-understandable representations through the mapping file nodes_peptides.csv, nodes_mhc.csv and nodes_tcr.csv:

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


