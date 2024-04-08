import copy
import datetime
import pdb
import random

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
import pickle
import os
import numpy as np

import config.config as config


class Dataset(object):
    def __init__(self, path):
        self.path = path
        self.graph_dataset = GraphDataset(path)
        # if self.graph_dataset has pm_data, pt_data, pmt_data, then load them
        # check if self.graph_datasett has pm_data attribute

        # set global random seed for train test split
        seed = config.seed 
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        
        if hasattr(self.graph_dataset, 'pm_data'):
            self.pm_data = np.array(self.graph_dataset.pm_data)
            self.pt_data = np.array(self.graph_dataset.pt_data)
            self.pmt_data = np.array(self.graph_dataset.pmt_data)
            self.pm_data_test = np.array(self.graph_dataset.pm_data_test)
            self.pt_data_test = np.array(self.graph_dataset.pt_data_test)
            self.pmt_data_test = np.array(self.graph_dataset.pmt_data_test)
        else:
            print('please regenerate graph data')
            exit()
        

        '''
        pos = 0
        neg = 0
        for item in self.pm_data:  # iterate all the item in pmt test
            if item[2] == 1:
                pos = pos +1
            else:
                neg = neg +1

        pos1 = 0
        neg1= 0
        for item in self.pt_data:  # iterate all the item in pmt test
            if item[2] == 1:
                pos1 = pos1 +1
            else:
                neg1 = neg1 +1
        '''
        
        
        self.pm_dataset = GeneralDataset(self.pm_data)
        self.pt_dataset = GeneralDataset(self.pt_data)
        self.pmt_dataset = GeneralDataset(self.pmt_data)
        self.pm_dataset_test = GeneralDataset(self.pm_data_test)
        self.pt_dataset_test = GeneralDataset(self.pt_data_test)
        self.pmt_dataset_test = GeneralDataset(self.pmt_data_test)
    
class GeneralDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class GraphDataset(InMemoryDataset):
    def __init__(self, path, transform=None, pre_transform=None):

        self.path = path 
        super().__init__(path, transform, pre_transform)
        print("load from processed data")
        self.graph_data, self.slices = torch.load(self.processed_paths[0])
        self.pm_data = torch.load(f"{self.path}/processed/pm_data.pt")
        self.pt_data = torch.load(f"{self.path}/processed/pt_data.pt")
        self.pmt_data = torch.load(f"{self.path}/processed/pmt_data.pt")
        self.pm_data_test = torch.load(f"{self.path}/processed/pm_data_test.pt")
        self.pt_data_test = torch.load(f"{self.path}/processed/pt_data_test.pt")
        self.pmt_data_test = torch.load(f"{self.path}/processed/pmt_data_test.pt")
        self.pt_m_dict = pickle.load(open(f"{self.path}/pt_m_dict.pkl", 'rb'))


    @property
    def raw_file_names(self):
        return ['{}/pm_data.pkl'.format(self.path),
                '{}/pt_data.pkl'.format(self.path),
                '{}/pmt_data.pkl'.format(self.path),
                '{}/statics.pkl'.format(self.path),
                '{}/pt_data_train.pkl'.format(self.path),
                '{}/pt_data_test.pkl'.format(self.path)
        ]

    @property
    def processed_file_names(self):
        return ['graph.dataset']


    def download(self):
        # Download to `self.raw_dir`.
        pass

    def read_data(self):

        try:
            self.pmt_data = pickle.load(open(self.pmt_file, 'rb'))
            self.pmt_data = np.array(self.pmt_data)
            self.pmt_data, self.pmt_data_test = train_test_split(self.pmt_data, test_size=config.test_size, random_state=config.seed)
        except:
            pmt_file_train = self.pmt_file.split('.pkl')[0]+'_train.pkl'
            pmt_file_test = self.pmt_file.split('.pkl')[0]+'_test.pkl'
            self.pmt_data = pickle.load(open(pmt_file_train, 'rb'))
            self.pmt_data_test = pickle.load(open(pmt_file_test, 'rb'))
            self.pmt_data = np.array(self.pmt_data)
            self.pmt_data_test = np.array(self.pmt_data_test)


        # use sklearn train test split to split data
        try:
            self.pm_data = pickle.load(open(self.pm_file, 'rb'))
            self.pm_data = np.array(self.pm_data)
            label = self.pm_data[:, 2]
            self.pm_data, self.pm_data_test = train_test_split(self.pm_data, test_size=config.test_size, random_state=config.seed)
        except:
            pm_file_train = self.pm_file.split('.pkl')[0]+'_train.pkl'
            pm_file_test = self.pm_file.split('.pkl')[0]+'_test.pkl'
            self.pm_data = pickle.load(open(pm_file_train, 'rb'))
            self.pm_data_test = pickle.load(open(pm_file_test, 'rb'))
            self.pm_data = np.array(self.pm_data)
            self.pm_data_test = np.array(self.pm_data_test)
        #self.pmt_data, self.pmt_data_test = train_test_split(self.pmt_data, test_size=config.test_size, random_state=config.seed)
        #label = self.pt_data[:, 0]
        #self.pt_data, self.pt_data_test = train_test_split(self.pt_data, test_size=config.test_size, random_state=42, stratify=label)
        try:
            self.pt_data = pickle.load(open(self.pt_file, 'rb'))
            self.pt_data = np.array(self.pt_data)
            label = self.pt_data[:, 2]
            self.pt_data, self.pt_data_test = train_test_split(self.pt_data, test_size=config.test_size, random_state=config.seed, stratify=label)
        except:
            self.pt_data = pickle.load(open(self.pt_file_train, 'rb'))
            self.pt_data_test = pickle.load(open(self.pt_file_test, 'rb'))
            #self.pt_data = self.pt_data + self.pt_data_test
            self.pt_data = np.array(self.pt_data)
            self.pt_data_test = np.array(self.pt_data_test)
            #splitter = CustomTrainTestSplit(data=self.pt_data, split_index=2) 
            #self.pt_data, self.pt_data_test = splitter.split()
        #splitter = CustomTrainTestSplit(data=self.pt_data, split_index=2) 
        #self.pt_data, self.pt_data_test = splitter.split()
        #self.pt_data, self.pt_data_test = train_test_split(self.pt_data, test_size=config.test_size, random_state=config.seed, stratify=label)

        
        if config.task == 'pt':
            # regen data for pt train/test, especially for pmt dataset
            if config.add_redundance:
                print("add redundance, regenerate pmt data train/test")
                pmt_list = []
                pmt_test_list = []
                time1 = datetime.datetime.now()
                all_pmt_data = np.concatenate((self.pmt_data, self.pmt_data_test), axis=0)
                for pmt in all_pmt_data:
                    if (np.all(np.array([pmt[0], pmt[2], 1]) == self.pt_data_test, axis=1)).sum() > 0:
                        pmt_test_list.append(pmt)
                    else:
                        pmt_list.append(pmt)

                self.pmt_data = np.array(pmt_list)
                self.pmt_data_test = np.array(pmt_test_list)
                    
                time2 = datetime.datetime.now()
                duration = time2 - time1
                print(duration) 
        elif config.task == 'pmt':
            # regen data for pmt train/test, especially for pt dataset
            if config.add_redundance:
                print("add redundance, regenerate pt data train/test")
                pt_list = []
                pt_test_list = []
                time1 = datetime.datetime.now()
                pmt_tmp = self.pmt_data_test[:, [0, 2]]
                all_pt_data = np.concatenate((self.pt_data, self.pt_data_test), axis=0)
                all_pt_data_pos = all_pt_data[all_pt_data[:, 2] == 1]
                all_pt_data_neg = all_pt_data[all_pt_data[:, 2] == 0]
                for pt in all_pt_data_pos:
                    if (np.all(np.array([pt[0], pt[1]]) == pmt_tmp, axis=1)).sum() > 0:
                        pt_test_list.append(pt)
                    else:  
                        pt_list.append(pt)
                self.pt_data = np.array(pt_list)
                self.pt_data_test = np.array(pt_test_list)

                neg_pt, neg_pt_test = train_test_split(all_pt_data_neg, test_size=config.test_size, random_state=config.seed)

                self.pt_data = np.concatenate((self.pt_data, neg_pt), axis=0)
                self.pt_data_test = np.concatenate((self.pt_data_test, neg_pt_test), axis=0)
                    
                time2 = datetime.datetime.now()
                duration = time2 - time1
                print(duration) 

        elif config.task == 'pm':
            # regen data for pm train/test, especially for pm dataset
            if config.add_redundance:
                print("add redundance, regenerate pmt data train/test")
                pmt_list = []
                pmt_test_list = []
                time1 = datetime.datetime.now()
                all_pmt_data = np.concatenate((self.pmt_data, self.pmt_data_test), axis=0)
                for pmt in all_pmt_data:
                    if (np.all(np.array([pmt[0], pmt[1]]) == self.pm_data_test[:, :2], axis=1)).sum() > 0: 
                        pmt_test_list.append(pmt)
                    else:
                        pmt_list.append(pmt)

                self.pmt_data = np.array(pmt_list)
                self.pmt_data_test = np.array(pmt_test_list)
                    
                time2 = datetime.datetime.now()
                duration = time2 - time1
                print(duration)
            self.pmt_data, self.pmt_data_test = train_test_split(self.pmt_data, test_size=config.test_size, random_state=config.seed)
            self.pt_data, self.pt_data_test = train_test_split(self.pt_data, test_size=config.test_size, random_state=config.seed)
        
            

        self.statics = pickle.load(open(self.statics_file, 'rb'))
        self.p_num = self.statics['p_num']
        self.t_num = self.statics['t_num']
        self.m_num = self.statics['m_num']

        # for test, select 10 data each

        print('p_num: {}, t_num: {}, m_num: {}'.format(self.p_num, self.t_num, self.m_num))

        self.total_node = self.p_num + self.t_num + self.m_num

        self.m_offset = 0
        self.t_offset = self.m_offset + self.m_num
        self.p_offset = self.t_offset + self.t_num

        print("construct graph")
        data = HeteroData()

        data['p'].x = torch.tensor(range(self.p_num))
        data['t'].x = torch.tensor(range(self.t_num))
        data['m'].x = torch.tensor(range(self.m_num))


        print('process pm data')
        pm_edges = [[], []]
        pm_set = set()
        if not config.ablate_pm:
            for triple in self.pm_data:
                if triple[2] == 1:
                    if (triple[0], triple[1]) in pm_set:
                        continue
                    pm_edges[0].append(triple[0])
                    pm_edges[1].append(triple[1])
                    pm_set.add((triple[0], triple[1]))
        
        if not config.ablate_pmt:
            for triple in self.pmt_data:
                if (triple[0], triple[1]) in pm_set:
                    continue
                pm_edges[0].append(triple[0])
                pm_edges[1].append(triple[1])
                pm_set.add((triple[0], triple[1]))

        data['p', 'pm_link', 'm'].edge_index = torch.tensor(pm_edges, dtype=torch.int64)

        print('process pt data')
        pt_edges = [[], []]
        if not config.ablate_pt:
            for triple in self.pt_data:
                if triple[2] == 1:
                    pt_edges[0].append(triple[0])
                    pt_edges[1].append(triple[1])
        
        data['p', 'pt_link', 't'].edge_index = torch.tensor(pt_edges, dtype=torch.int64)


        print('process mt data')
        mt_edges = [[], []]
        if not config.ablate_pmt:
            for triple in self.pmt_data:
                mt_edges[0].append(triple[1])
                mt_edges[1].append(triple[2])
        
        data['m', 'mt_link', 't'].edge_index = torch.tensor(mt_edges, dtype=torch.int64)

        return data
        

    def process(self):
        if not os.path.exists(f"{self.path}/processed"):
            os.mkdir(f"{self.path}/processed")

        print("read data")
        self.pm_file = self.raw_file_names[0]
        self.pt_file = self.raw_file_names[1]
        self.pmt_file = self.raw_file_names[2]
        self.statics_file = self.raw_file_names[3]
        self.pt_file_train = self.raw_file_names[4]
        self.pt_file_test = self.raw_file_names[5]

        print("process data")
        self.graph = self.read_data()
        self.graph = T.ToUndirected()(self.graph)


        print("save data")
        graph, slices = self.collate([self.graph])
        print(self.processed_paths)
        torch.save((graph, slices), self.processed_paths[0])
        #torch.save(self.graphs, self.processed_paths[0])
        # save self.pm_data
        torch.save(self.pm_data, f"{self.path}/processed/pm_data.pt")
        torch.save(self.pt_data, f"{self.path}/processed/pt_data.pt")
        torch.save(self.pmt_data, f"{self.path}/processed/pmt_data.pt")
        torch.save(self.pm_data_test, f"{self.path}/processed/pm_data_test.pt")
        torch.save(self.pt_data_test, f"{self.path}/processed/pt_data_test.pt")
        torch.save(self.pmt_data_test, f"{self.path}/processed/pmt_data_test.pt")

    def feature_N(self):
        return self.feature_num

    def data_N(self):
        return self.data_num


class CustomTrainTestSplit:
    def __init__(self, data, split_index, test_size=config.test_size, random_state=config.seed):
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.split_index = split_index

    def split(self):
        split_info = self.data[:, self.split_index]
        unique_classes, class_counts = np.unique(split_info, return_counts=True)

        single_sample_classes = unique_classes[class_counts <= 3]
        single_sample_indices = np.isin(split_info, single_sample_classes)
        single_sample_data = self.data[single_sample_indices]
        remaining_data = self.data[~single_sample_indices]

        remaining_split_info = remaining_data[:, self.split_index]
        train_data, test_data = train_test_split(
            remaining_data, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=remaining_split_info
        )

        train_data = np.vstack([train_data, single_sample_data])

        return train_data, test_data