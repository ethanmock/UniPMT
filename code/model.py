import pdb

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import config.config as config


class MolGNN(nn.Module):

    def __init__(self, path, p_num, t_num, m_num, hidden_size, meta_data, device):
        super(MolGNN, self).__init__()
        self.m_num = m_num
        self.p_num = p_num
        self.t_num = t_num
        self.device = device
        self.p_Embedding = torch.from_numpy(np.load(f'{path}/p_features.npy')).float().to(self.device)
        self.t_Embedding = torch.from_numpy(np.load(f'{path}/t_features.npy')).float().to(self.device)

        self.m_Embedding_meta = torch.from_numpy(np.load(f'{path}/m_features.npy')).float().to(self.device)
        m_size = self.m_Embedding_meta.shape[1]
        emb_size = config.gnn_out_emb_size

        self.gnn_model = GNN(config.hidden_size, emb_size)
        self.gnn_model = to_hetero(self.gnn_model, meta_data, aggr='mean').to(self.device)

        self.m_emb_model = nn.Sequential(
            nn.Linear(m_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, config.emb_size_hid)
        )

        # deprecated
        self.pm_model = nn.Sequential(
            #nn.LayerNorm(config.gnn_out_emb_size * 2),
            #nn.BatchNorm1d(config.gnn_out_emb_size * 2),
            nn.Linear(config.gnn_out_emb_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            #nn.LayerNorm(hidden_size),
            nn.Dropout(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            #nn.BatchNorm1d(hidden_size),
            #nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            #nn.LayerNorm(hidden_size),
            nn.Dropout(),
            nn.Linear(hidden_size, config.out_emb_size),
            )

        self.pm_part1 = self.pm_block(config.gnn_out_emb_size, hidden_size)
        self.pm_part2 = self.pm_block(hidden_size, hidden_size)
        self.pm_part3 = self.pm_block(hidden_size, hidden_size)
        self.pm_part4 = self.pm_block(hidden_size, hidden_size)
        self.act1 = nn.LayerNorm(hidden_size)
        self.act2 = nn.LayerNorm(hidden_size)
        self.act3 = nn.LayerNorm(hidden_size)
        self.act4 = nn.LayerNorm(hidden_size)
        self.linear_pm = nn.Linear(config.gnn_out_emb_size * 2, config.gnn_out_emb_size)

        # mt model
        self.mt_model = nn.Sequential(
            nn.Linear(config.gnn_out_emb_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(),
            nn.Linear(hidden_size, config.out_emb_size)
        )

        self.cross_model = nn.Sequential(
            nn.Linear(config.out_emb_size, hidden_size),
            nn.Linear(hidden_size, 1)
        )
        self.w = nn.Parameter(torch.rand(config.out_emb_size, 1))
        self.norm = nn.LayerNorm(config.gnn_out_emb_size)

    def pm_block(self, in_size, hidden_size):
        return nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
        )

    def gnn_learn(self, x, weight=1):
        x = x.to(self.device)
        emb_dict = {}
        emb_dict['p'] = self.p_Embedding[x['p'].x]
        self.m_Embedding = self.m_emb_model(self.m_Embedding_meta)
        emb_dict['m'] = self.m_Embedding[x['m'].x]
        emb_dict['t'] = self.t_Embedding[x['t'].x]
        if config.task == 'pm':
            out = {}
            out['p'] = emb_dict['p']
            out['m'] = emb_dict['m']
            out['t'] = emb_dict['t']
        elif config.task =='pmt':
            weight = 0.01 #0.001 0.79; 0.0 0.79;0.01 0.71; 0.1 0.65
            gnn_out = self.gnn_model(emb_dict, x.edge_index_dict)
            # perform add and norm on the output of emb_dict and out
            out = {}
            out['p'] = weight*gnn_out['p'] + emb_dict['p']
            out['m'] = weight*gnn_out['m'] + emb_dict['m']
            out['t'] = weight*gnn_out['t'] + emb_dict['t']
        else:
            weight = 1#zero 0: 55, 74; gnnonly: 53,73; 1 is the best; teim
            gnn_out = self.gnn_model(emb_dict, x.edge_index_dict)
            # perform add and norm on the output of emb_dict and out
            out = {}
            out['p'] = weight*gnn_out['p'] + emb_dict['p']
            out['m'] = weight*gnn_out['m'] + emb_dict['m']
            out['t'] = weight*gnn_out['t'] + emb_dict['t']
        return out

    def pm_pred(self, p_gnn_emb, m_gnn_emb):

        pm = self.pm_learn(p_gnn_emb, m_gnn_emb)

        pred = torch.matmul(pm, self.w)
        pred = F.sigmoid(pred)

        return pred

    def pm_learn(self, p_emb, m_emb):
        
        x = torch.cat((p_emb, m_emb), dim=1)
        res = self.linear_pm(x)
        res = self.act1(res)
        out = self.pm_part1(res)
        res = self.act1(out + res)
        out = self.pm_part2(res)
        res = self.act1(out + res)
        out = self.pm_part3(res)
        res = self.act1(out + res)
        out = self.pm_part4(res)
        res = self.act1(out + res)
        return res

    def mt_learn(self, m_emb, t_emb):
        x = torch.cat((m_emb, t_emb), dim=1)
        out = self.mt_model(x)
        return out

    def pmt_cross(self, pm_out, mt_out):
        # do the dot product of pm_out and mt_out and use sigmoid to get the probability
        inp = pm_out * mt_out

        out = self.cross_model(inp)
        out = F.sigmoid(out)
        return out

    def pmt_learn(self, p_out_emb, m_out_emb, t_out_emb, pmt_batch):

        p_emb = p_out_emb[pmt_batch[:, 0]]
        m_emb = m_out_emb[pmt_batch[:, 1]]
        t_emb = t_out_emb[pmt_batch[:, 2]]

        n_sample = pmt_batch[:, 0].shape[0]

        pm_pos = self.pm_learn(p_emb, m_emb)
        mt_pos = self.mt_learn(m_emb, t_emb)
        pmt_pos = self.pmt_cross(pm_pos, mt_pos)

        pmt_neg = []

        for i in range(config.pmt_run_negative):
            random_indexs_m = torch.randint(0, self.m_num, (n_sample,))
            random_indexs_t = torch.randint(0, self.t_num, (n_sample,))

            p_emb_neg = p_emb
            m_emb_neg = m_out_emb[random_indexs_m]
            t_emb_neg = t_out_emb[random_indexs_t]

            pm_neg = self.pm_learn(p_emb_neg, m_emb_neg)
            mt_neg = self.mt_learn(m_emb_neg, t_emb_neg)

            # get all the permutation of pmt_neg
            pmt_neg.append(self.pmt_cross(pm_neg, mt_neg))

        return pmt_pos, pmt_neg

    def pt_pred(self, p_emb, t_emb, m_emb):

        pm_out = self.pm_learn(p_emb, m_emb)
        mt_out = self.mt_learn(m_emb, t_emb)

        out = self.pmt_cross(pm_out, mt_out)

        return out

    def neg_sampling(self, emb):
        # permute the emb
        emb = emb[torch.randperm(emb.size()[0])]
        return emb


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv2 = SAGEConv((-1, -1), out_channels, aggr='add')
        self.act = nn.Tanh()
        

    def forward(self, x, edge_index):
        gnn_out = self.conv2(x, edge_index)
        gnn_out = self.act(gnn_out)
        out = gnn_out
        return out 