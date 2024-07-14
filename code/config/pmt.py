m_weight = 0.001#0.001
pt_weight = 0.5#0.5
pmt_weight = 0.1#0.1

pt_neg_sample = 1
lr = 0.000004
reg_lambda = 0.00001

epoch_num = 10
#PMT AUC: 0.9521, PMT PRAUC: 0.7728
ablate_pm = False #PMT AUC: 0.5120, PMT PRAUC: 0.0914
ablate_pt = False #PMT AUC: 0.9236, PMT PRAUC: 0.6892
ablate_pmt = False #PMT AUC: 0.8402, PMT PRAUC: 0.4440
pmt_run_negative = 10
sample_m = False

# train around 100 epoch to get the reported results
