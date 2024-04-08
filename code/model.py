import pdb

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero, GATConv, GINConv, GCNConv
from torch_geometric.nn import TransformerConv

import config.config as config



# a simple molgnn model with only one layer MLP
class MolGNN(nn.Module):
    
    def __init__(self,path, p_num, t_num, m_num, hidden_size, meta_data, device): 

        super(MolGNN, self).__init__()

        self.m_num = m_num
        self.p_num = p_num
        self.t_num = t_num
        self.device = device

        #self.p_Embedding = nn.Embedding(p_num, emb_size)
        #self.m_Embedding = nn.Embedding(m_num, emb_size) 
        #self.t_Embedding = nn.Embedding(t_num, emb_size)
        # load numpy files to tensor
        self.p_Embedding = torch.from_numpy(np.load(f'{path}/p_features.npy')).float().to(self.device)
        self.t_Embedding = torch.from_numpy(np.load(f'{path}/t_features.npy')).float().to(self.device)

        self.m_Embedding_meta = torch.from_numpy(np.load(f'{path}/m_features.npy')).float().to(self.device)
        m_size = self.m_Embedding_meta.shape[1]
        emb_size = config.gnn_out_emb_size
        #self.m_emb_model = nn.Linear(m_size, emb_size).to(device)
        #self.m_Embedding = self.m_emb_model(self.m_Embedding_meta)


        self.gnn_model = GNN(config.hidden_size, emb_size)
        self.gnn_model = to_hetero(self.gnn_model, meta_data, aggr='mean').to(self.device)

        self.m_emb_model = nn.Sequential(
            nn.Linear(m_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            #nn.Dropout(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            #nn.BatchNorm1d(hidden_size),
            #nn.Dropout(),
            nn.Linear(hidden_size, config.emb_size_hid)
            )

        """
        self.p_emb_model = nn.Sequential(nn.Linear(emb_size, hidden_size),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(hidden_size),
                                      #nn.Dropout(),
                                      nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(hidden_size),
                                      #nn.Dropout(),
                                      nn.Linear(hidden_size, config.emb_size_hid)
                                      )
        self.t_emb_model = nn.Sequential(nn.Linear(emb_size, hidden_size),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(hidden_size),
                                      #nn.Dropout(),
                                      nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(hidden_size),
                                      #nn.Dropout(),
                                      nn.Linear(hidden_size, config.emb_size_hid)
                                      )
        """
        # pm model
        self.pm_model = nn.Sequential(
            nn.Linear(config.gnn_out_emb_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            #nn.BatchNorm1d(hidden_size),
            #nn.Dropout(),
            nn.Linear(hidden_size, config.out_emb_size)
            )

        # mt model
        self.mt_model = nn.Sequential(
            nn.Linear(config.gnn_out_emb_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            #nn.BatchNorm1d(hidden_size),
            #nn.Dropout(),
            nn.Linear(hidden_size, config.out_emb_size)
            )

        
        self.cross_model = nn.Sequential(
            nn.Linear(config.out_emb_size, hidden_size),
            #nn.ReLU(),
            #nn.BatchNorm1d(hidden_size),
            #nn.Dropout(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            #nn.BatchNorm1d(hidden_size),
            #nn.Dropout(),
            nn.Linear(hidden_size, 1)
            #nn.Linear(config.out_emb_size, 1)
            )
        self.w = nn.Parameter(torch.rand(config.out_emb_size, 1))
    
        

    
    def gnn_learn(self, x, check):
        x = x.to(self.device)
        emb_dict = {}
        emb_dict['p'] = self.p_Embedding[x['p'].x]
        self.m_Embedding = self.m_emb_model(self.m_Embedding_meta)
        emb_dict['m'] = self.m_Embedding[x['m'].x]
        emb_dict['t'] = self.t_Embedding[x['t'].x]
        out = self.gnn_model(emb_dict, x.edge_index_dict)
        if check:
            pdb.set_trace()
        return out
        
    def pm_pred(self, p_emb, m_emb):

        pm = self.pm_learn(p_emb, m_emb)

        pred = torch.matmul(pm, self.w) 
        pred = F.sigmoid(pred)

        return pred

    def pm_learn(self, p_emb, m_emb):
        x = torch.cat((p_emb, m_emb), dim=1)
        out = self.pm_model(x)
        return out
    
    def mt_learn(self, m_emb, t_emb):
        x = torch.cat((m_emb, t_emb), dim=1)
        out = self.mt_model(x)
        return out
    
    def pmt_cross(self, pm_out, mt_out):
        # do the dot product of pm_out and mt_out and use sigmoid to get the probability
        inp = pm_out * mt_out

        out = self.cross_model(inp)
        #out = (pm_out * mt_out).sum(dim=1)
        #out = F.sigmoid(out) ** (1/3)
        out = F.sigmoid(out)
        return out
    
    
    def pmt_learn(self, p_out_emb, m_out_emb, t_out_emb, pmt_batch):

        p_emb = p_out_emb[pmt_batch[:, 0]]
        m_emb = m_out_emb[pmt_batch[:, 1]]
        t_emb = t_out_emb[pmt_batch[:, 2]]

        n_sample = pmt_batch[:, 0].shape[0]

        pm_pos = self.pm_learn(p_emb, m_emb)
        mt_pos = self.mt_learn(m_emb, t_emb)
        pmt_pos =  self.pmt_cross(pm_pos, mt_pos)

        pmt_neg = []

        for i in range(20):
            random_indexs_p = torch.randint(0, self.p_num, (n_sample,))
            random_indexs_m = torch.randint(0, self.m_num, (n_sample,)) 
            random_indexs_t = torch.randint(0, self.t_num, (n_sample,))
            
            p_emb_neg = p_out_emb[random_indexs_p]
            m_emb_neg = m_out_emb[random_indexs_m]
            t_emb_neg = t_out_emb[random_indexs_t]

            pm_neg = self.pm_learn(p_emb_neg, m_emb_neg)
            mt_neg = self.mt_learn(m_emb_neg, t_emb_neg)

            # get all the permutation of pmt_neg
            pmt_neg.append(self.pmt_cross(pm_neg, mt_neg))

        return pmt_pos, pmt_neg

        
    def pt_pred(self, p_emb, t_emb, m_emb):
        # select indices of max value of len(self.e_emb.shape[0])
        # and length of p_emb.shape[0]
        #m_gnn_emb_meta = self.m_Embedding_meta[indices_m]
        #m_emb = self.m_emb_model(m_gnn_emb_meta)
        
        pm_out = self.pm_learn(p_emb, m_emb)
        mt_out = self.mt_learn(m_emb, t_emb)

        out = self.pmt_cross(pm_out, mt_out)

        return out

        
    def neg_sampling(self, emb):
        # permute the emb
        emb = emb[torch.randperm(emb.size()[0])]
        return emb
    

"""        
class PM_Model(nn.Module):
    
    def __init__(self, emb_size, hidden_size):
        super(PM_Model, self).__init__()
        self.fc1 = nn.Linear(emb_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, p_emb, m_emb):
        x = torch.cat((p_emb, m_emb), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    
class PMT_Model(nn.Module):
    #PMT model is a contrastive model that give p and m, we contrast different t
    
    def __init__(self, emb_size, hidden_size, pm_model):
        super(PMT_Model, self).__init__()


        self.pm_model = pm_model
    
    def neg_sample(self, p_emb, m_emb, t_emb):
        # shuffle p_emb, m_emb, t_emb
        p_neg = torch.randperm(p_emb.shape[0])
        m_neg = torch.randperm(m_emb.shape[0])
    
    def forward(self, p_emb, m_emb, t_emb):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    

class PT_Model(nn.Module):
    
    def __init__(self, emb_size, hidden_size):
        super(PT_Model, self).__init__()
        self.fc1 = nn.Linear(emb_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, p_emb, t_emb):
        x = torch.cat((p_emb, t_emb), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
"""

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        #self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels, aggr='mean')
        #self.conv2 = GCNConv((-1, -1), out_channels, add_self_loops=False)
        #self.conv1 = TransformerConv((-1, -1), hidden_channels, add_self_loops=False)
        #self.conv2 = TransformerConv((-1, -1), out_channels, add_self_loops=False)
        #self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        # gin model
        """
        self.gin_model = nn.Sequential(
            nn.Linear(config.emb_size_hid, hidden_channels),
            nn.ReLU(),
            #nn.BatchNorm1d(hidden_channels),
            #nn.Dropout(),
            #nn.Linear(hidden_channels, hidden_channels),
            #nn.ReLU(),
            #nn.BatchNorm1d(hidden_channels),
            #nn.Dropout(),
            nn.Linear(hidden_channels, out_channels) 
            )
        self.conv2 = GINConv(nn=self.gin_model)
        """

    def forward(self, x, edge_index):
        #x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x