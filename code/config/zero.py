pm_weight = 0.1#0.001
pt_weight = 0.1
pmt_weight = 0.1
pmt_run_negative = 1

lr = 0.000004
reg_lambda = 0.00001
# If changed, should rerun data_preprocess.py
pt_neg_sample = 1
sample_m = False

batch_size = 512

ablate_pm = False
ablate_pt = False
ablate_pmt = False
# train around 100 epoch to get the reported results