import pdb

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
import numpy as np
from torch_scatter import scatter_mean

from model import MolGNN
import config.config as config


class Tester():
    
    def __init__(self, path, data):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = 'cpu'
        print(f'Using device: {self.device}')

        #self.graph = data.graph_dataset.graph_data.to(self.device)
        self.graph = data.graph_dataset.graph_data
        self.p_num = self.graph['p'].x.shape[0]
        self.t_num = self.graph['t'].x.shape[0] 
        self.m_num = self.graph['m'].x.shape[0]
        self.pt_m_dict = data.graph_dataset.pt_m_dict
        meta = self.graph.metadata()


        pm_dataset = data.pm_dataset
        pt_dataset = data.pt_dataset
        pmt_dataset = data.pmt_dataset

        pm_dataset_test = data.pm_dataset_test
        pt_dataset_test = data.pt_dataset_test
        pmt_dataset_test = data.pmt_dataset_test
        

        pm_data_len = len(pm_dataset)
        pt_data_len = len(pt_dataset)
        pmt_data_len = len(pmt_dataset)
        n_batch = pmt_data_len // config.batch_size 
        pm_batch = (pm_data_len // n_batch) 
        pt_batch = (pt_data_len // n_batch) 

        self.pm_loader_test = DataLoader(pm_dataset_test, batch_size=pm_batch, shuffle=False, num_workers=4)
        self.pt_loader_test = DataLoader(pt_dataset_test, batch_size=pt_batch, shuffle=False, num_workers=4)
        self.pmt_loader_test = DataLoader(pmt_dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=4)

        self.model = MolGNN(path, self.p_num, self.t_num, self.m_num, config.hidden_size, meta, self.device)
        self.model.to(self.device)


    def pm_predict(self, pm_batch):
        p_gnn_emb = self.p_out_emb[pm_batch[:, 0]]
        m_gnn_emb = self.m_out_emb[pm_batch[:, 1]]
        label = pm_batch[:, 2].float().to(self.device).unsqueeze(-1)

        pm_pred = self.model.pm_pred(p_gnn_emb, m_gnn_emb)
        return pm_pred, label
    
        
    def pt_predict(self, pt_batch):
        leaveout_indices = []
        p_indices = []
        t_indices = []
        m_indices = []
        for i, pt_sample in enumerate(pt_batch):
            tup = tuple(pt_sample[:2].tolist())
            if pt_sample[2] == 1 and tup in self.pt_m_dict:
                p_indices.append(pt_sample[0])
                t_indices.append(pt_sample[1])
                m_indices.append(np.random.choice(list(self.pt_m_dict[tup])))
                leaveout_indices.append(i)
            else:
                p_indices += [pt_sample[0]] * self.m_num
                t_indices += [pt_sample[1]] * self.m_num
                m_indices += list(range(self.m_num))
                leaveout_indices += [i] * self.m_num
        
        p_indices = torch.tensor(p_indices).to(self.device)
        t_indices = torch.tensor(t_indices).to(self.device)
        m_indices = torch.tensor(m_indices).to(self.device)
        p_gnn_emb = self.p_out_emb[p_indices]
        t_gnn_emb = self.t_out_emb[t_indices]
        m_gnn_emb = self.m_out_emb[m_indices]
        
        pt_pred_flat = self.model.pt_pred(p_gnn_emb, t_gnn_emb, m_gnn_emb).squeeze()
        label = pt_batch[:, 2].float().to(self.device)
        pt_pred = scatter_mean(pt_pred_flat, torch.tensor(leaveout_indices).to(self.device), dim=0)
            
        return pt_pred, label
        
        

    def pmt_predict(self, pmt_batch):
        pmt_pos, pmt_neg_list = self.model.pmt_learn(self.p_out_emb, self.m_out_emb, self.t_out_emb, pmt_batch)
        # get the contrastive loss of positive and negative samples
        pmt_pos = pmt_pos.unsqueeze(0).squeeze(-1)
        pmt_negs = torch.stack(pmt_neg_list).squeeze(-1)

        return pmt_pos, pmt_negs
        

    def evaluate(self, epoch, evaltask=['pm', 'pt', 'pmt']):

        #print("Start evaluating...")
        self.model.eval()

        if 'pm' in evaltask:
            pm_pred_list = []
            label_list_pm = []
            for pm_batch in self.pm_loader_test:
                pm_pred, label = self.pm_predict(pm_batch)
                pm_pred_list += pm_pred.tolist()
                label_list_pm += label.tolist()
            pm_pred_list = np.array(pm_pred_list).squeeze()
            label_list_pm = np.array(label_list_pm).squeeze()
            pm_auc = roc_auc_score(label_list_pm, pm_pred_list) 
            y_pred_label = [1 if prob >= config.pm_threshold else 0 for prob in pm_pred_list]
            pm_acc = accuracy_score(label_list_pm, y_pred_label)

            pm_pre, pm_rec, _= precision_recall_curve(label_list_pm, pm_pred_list)
            pm_prauc = auc(pm_rec,pm_pre)
            print(f"PM AUC: {pm_auc:.4f}, PM PRAUC: {pm_prauc:.4f}", end=' ')

            df = pd.DataFrame()
            df['prob'] = pm_pred_list
            df['label'] = label_list_pm
            pred_path = f"../output/predict/result_pm_{config.dataname}.csv"
            df.to_csv(pred_path, index=False)

        if 'pt' in evaltask:
            pt_pred_list = []
            label_list_pt = []
            p_list = []
            for pt_batch in self.pt_loader_test:
                pt_pred, label = self.pt_predict(pt_batch)
                pt_pred_list += pt_pred.tolist()
                label_list_pt += label.tolist()
                p_list += pt_batch[:, 0].tolist()

            pt_pred_list = np.array(pt_pred_list).squeeze()
            label_list_pt = np.array(label_list_pt).squeeze()
            p_list = np.array(p_list).squeeze()
            # pt_auc = self.cal_auc_by_group(p_list, pt_pred_list, label_list_pt)
            pt_auc = roc_auc_score(label_list_pt, pt_pred_list)
            y_pred_label = [1 if prob >= config.pt_threshold else 0 for prob in pt_pred_list]
            pt_acc = accuracy_score(label_list_pt, y_pred_label)
            pt_pre, pt_rec, _ = precision_recall_curve(label_list_pt, pt_pred_list)
            pt_prauc = auc(pt_rec, pt_pre)
            print(f"PT AUC: {pt_auc:.4f}, PT PRAUC: {pt_prauc:.4f}", end=' ')

            df = pd.DataFrame()
            df['prob'] = pt_pred_list
            df['label'] = label_list_pt
            pred_path = f"../output/predict/result_pt_{config.dataname}.csv"
            df.to_csv(pred_path, index=False)

        if 'pmt' in evaltask:
            pmt_pos_list = []
            pmt_neg_list = []
            for pmt_batch in self.pmt_loader_test:
                pmt_pos, pmt_negs = self.pmt_predict(pmt_batch)
                pmt_pos_list += pmt_pos
                pmt_neg_list += pmt_negs

            pmt_pos_all = torch.concat(pmt_pos_list).cpu().detach().numpy()
            pmt_neg_all = torch.concat(pmt_neg_list).cpu().detach().numpy()
            all_preds = np.concatenate([pmt_pos_all, pmt_neg_all])
            all_labels = np.concatenate([np.ones(pmt_pos_all.shape[0]), np.zeros(pmt_neg_all.shape[0])])
            pmt_auc = roc_auc_score(all_labels, all_preds)
            y_pred_label = [1 if prob >= config.pmt_threshold else 0 for prob in all_preds]
            pmt_acc = accuracy_score(all_labels, y_pred_label)

            pmt_pre, pmt_rec, pmt_thr=precision_recall_curve(all_labels, all_preds)
            pmt_prauc = auc(pmt_rec,pmt_pre)
            print(f"PMT AUC: {pmt_auc:.4f}, PMT PRAUC: {pmt_prauc:.4f}")
            print()

            df = pd.DataFrame()
            df['prob'] = all_preds
            df['label'] = all_labels
            pred_path = f"../output/predict/result_pmt_{config.dataname}.csv"
            df.to_csv(pred_path, index=False)
        
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

        gnn_out = self.model.gnn_learn(self.graph, False)
        self.p_out_emb = gnn_out['p']
        self.m_out_emb = gnn_out['m']
        self.t_out_emb = gnn_out['t']
