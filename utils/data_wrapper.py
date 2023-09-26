from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class Wrap_Dataset_fullInfo(Dataset):
    """Wrapper, convert <doc_feature, relevance_label> Tensor into Pytorch Dataset"""
    def __init__(self, doc_tensor, label_tensor,  use_cuda=True):
        if use_cuda:
            self.label_tensor = label_tensor.cuda()
            self.doc_tensor = doc_tensor.cuda()
        else:
            self.label_tensor = label_tensor.cpu()
            self.doc_tensor = doc_tensor.cpu()

    def __getitem__(self, index):
        return self.doc_tensor[index],self.label_tensor[index]

    def __len__(self):
        return self.doc_tensor.size(0)

    
class Wrap_Dataset_clickLog(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, doc_tensor, click_tensor, ips_tensor,  use_cuda=True):
        if use_cuda:
            self.click_tensor = click_tensor.cuda()
            self.doc_tensor = doc_tensor.cuda()
            self.ips_tensor = ips_tensor.cuda()
        else:
            self.click_tensor = click_tensor.cpu()
            self.doc_tensor = doc_tensor.cpu()
            self.ips_tensor = ips_tensor.cpu()


    def __getitem__(self, index):
        return self.doc_tensor[index],self.click_tensor[index], self.ips_tensor[index]

    def __len__(self):
        return self.doc_tensor.size(0)


class Wrap_Dataset_clickLog2(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, doc_tensor, click_tensor, select_tensor, ips_tensor,  use_cuda=True):
        if use_cuda:
            self.click_tensor = click_tensor.cuda()
            self.select_tensor = select_tensor.cuda()
            self.doc_tensor = doc_tensor.cuda()
            self.ips_tensor = ips_tensor.cuda()
        else:
            self.click_tensor = click_tensor.cpu()
            self.select_tensor = select_tensor.cpu()
            self.doc_tensor = doc_tensor.cpu()
            self.ips_tensor = ips_tensor.cpu()


    def __getitem__(self, index):
        return self.doc_tensor[index],self.click_tensor[index], self.select_tensor[index], self.ips_tensor[index]

    def __len__(self):
        return self.doc_tensor.size(0)


class Wrap_Dataset_Pairwise2(Dataset):
    """Wrapper, convert <posi_feature, neg_feature, rel_dff> Tensor into Pytorch Dataset"""
    def __init__(self, posi_id, neg_id, rel_diff, dataset, use_cuda=True, sparse_tag=False):
        if use_cuda:
            self.posi_id = torch.LongTensor(posi_id).cuda()
            self.neg_id = torch.LongTensor(neg_id).cuda()
            self.rel_diff = torch.Tensor(rel_diff).cuda()
            unique_dataset = dataset[['did','feature']].drop_duplicates('did')
            if sparse_tag:
                # self.fes = self._get_sparse_feature(unique_dataset['feature'].tolist()).cuda()
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cuda()
            else:
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cuda()               
            self.didx = torch.LongTensor(np.array(unique_dataset['did'].tolist())).cuda()
            
        else:
            self.posi_id = torch.LongTensor(posi_id).cpu()
            self.neg_id = torch.LongTensor(neg_id).cpu()
            self.rel_diff = torch.Tensor(rel_diff).cpu()
            unique_dataset = dataset[['did','feature']].drop_duplicates('did')
            if sparse_tag:
                # self.fes = self._get_sparse_feature(unique_dataset['feature'].tolist()).cpu()
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cpu() 
            else:
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cpu()                
            self.didx = torch.LongTensor(np.array(unique_dataset['did'].tolist())).cpu()
        self.use_cuda = use_cuda
        self.sparse_tag = sparse_tag

    def _get_sparse_feature_forOnerow(self, row):
        idx = np.array(row, dtype=int)[:,0] - 1      
        val = np.array(row, dtype=float)[:,1]
        if 699 not in idx:
            idx = np.append(idx, 699) 
            val = np.append(val, 0.)
        sparse_fe = torch.sparse.FloatTensor(torch.LongTensor([idx]), torch.FloatTensor(val))
        return sparse_fe
    
    def _get_sparse_feature(self,dat):
        row_idx_ls = []
        col_idx_ls = []
        val_ls = []
        # dat = dat.apply(pre_trans_row)
        for i in range(len(dat)):
            row_idx_ls.extend([i]*len(dat[i])) 
            col_idx_ls.extend(np.array(dat[i], dtype=int)[:,0] - 1)
            val_ls.extend(np.array(dat[i], dtype=float)[:,1])
        sparse_ts = torch.sparse.FloatTensor(torch.LongTensor([row_idx_ls, col_idx_ls]), torch.FloatTensor(val_ls)).to_dense()
        return sparse_ts
        
    def __getitem__(self, index):
        posi_feature = self.fes[torch.nonzero(self.didx==self.posi_id[index])[0]]
        neg_feature = self.fes[torch.nonzero(self.didx==self.neg_id[index])[0]]
        return posi_feature, neg_feature, self.rel_diff[index]

    def __len__(self):
        return len(self.posi_id) 
    

class Wrap_Dataset_Pairwise3(Dataset):
    """Wrapper, convert <posi_feature, neg_feature, rel_dff> Tensor into Pytorch Dataset"""
    def __init__(self, posi_id, neg_id, posi_pos, neg_pos, rel_diff, dataset, use_cuda=True, sparse_tag=False):
        if use_cuda:
            self.posi_id = torch.LongTensor(posi_id).cuda()
            self.neg_id = torch.LongTensor(neg_id).cuda()
            self.posi_pos = torch.LongTensor([posi_pos]).cuda()
            self.neg_pos = torch.LongTensor([neg_pos]).cuda()
            self.rel_diff = torch.Tensor(rel_diff).cuda()
            self.topK = dataset['neg_pos'].max()
            unique_dataset = dataset[['did','feature']].drop_duplicates('did')
            if sparse_tag:
                self.fes = self._get_sparse_feature(unique_dataset['feature'].tolist()).cuda()
            else:
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cuda()               
            self.didx = torch.LongTensor(np.array(unique_dataset['did'].tolist())).cuda()
            
        else:
            self.posi_id = torch.LongTensor(posi_id).cpu()
            self.neg_id = torch.LongTensor(neg_id).cpu()
            self.posi_pos = torch.LongTensor([posi_pos]).cpu()
            self.neg_pos = torch.LongTensor([neg_pos]).cpu()
            self.rel_diff = torch.Tensor(rel_diff).cpu()
            self.topK = dataset['neg_pos'].max()
            unique_dataset = dataset[['did','feature']].drop_duplicates('did')
            if sparse_tag:
                self.fes = self._get_sparse_feature(unique_dataset['feature'].tolist()).cpu()
            else:
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cpu()                
            self.didx = torch.LongTensor(np.array(unique_dataset['did'].tolist())).cpu()
        self.use_cuda = use_cuda
        self.sparse_tag = sparse_tag

    def _get_sparse_feature_forOnerow(self, row):
        idx = np.array(row, dtype=int)[:,0] - 1      
        val = np.array(row, dtype=float)[:,1]
        if 699 not in idx:
            idx = np.append(idx, 699) 
            val = np.append(val, 0.)
        sparse_fe = torch.sparse.FloatTensor(torch.LongTensor([idx]), torch.FloatTensor(val))
        return sparse_fe
    
    def _get_sparse_feature(self,dat):
        row_idx_ls = []
        col_idx_ls = []
        val_ls = []
        # dat = dat.apply(pre_trans_row)
        for i in range(len(dat)):
            row_idx_ls.extend([i]*len(dat[i])) 
            col_idx_ls.extend(np.array(dat[i], dtype=int)[:,0] - 1)
            val_ls.extend(np.array(dat[i], dtype=float)[:,1])
        sparse_ts = torch.sparse.FloatTensor(torch.LongTensor([row_idx_ls, col_idx_ls]), torch.FloatTensor(val_ls))
        return sparse_ts
        
    def __getitem__(self, index):
        posi_feature = self.fes[torch.nonzero(self.didx==self.posi_id[index])[0]]
        posi_position = torch.zeros(self.topK).scatter_(dim=0, index=self.posi_pos,value=1)
        neg_feature = self.fes[torch.nonzero(self.didx==self.neg_id[index])[0]]
        neg_position = torch.zeros(self.topK).scatter_(dim=0, index=self.neg_pos,value=1)
        return posi_feature, neg_feature, self.rel_diff[index], posi_position, neg_position

    def __len__(self):
        return len(self.posi_id) 


class Wrap_Dataset_Pairwise4(Dataset):
    """Wrapper, convert <posi_feature, neg_feature, rel_dff> Tensor into Pytorch Dataset"""
    def __init__(self, posi_id, neg_id, posi_sel, neg_sel, rel_diff, tag, dataset, use_cuda=True, sparse_tag=False):
        if use_cuda:
            self.posi_id = torch.LongTensor(posi_id).cuda()
            self.neg_id = torch.LongTensor(neg_id).cuda()
            self.rel_diff = torch.Tensor(rel_diff).cuda()
            self.posi_sel = torch.Tensor(posi_sel).cuda()
            self.neg_sel = torch.Tensor(neg_sel).cuda()
            self.tag = torch.LongTensor(tag).cuda()
            unique_dataset = dataset[['did','feature']].drop_duplicates('did')
            if sparse_tag:
                # self.fes = self._get_sparse_feature(unique_dataset['feature'].tolist()).cuda()
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cuda()
            else:
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cuda()               
            self.didx = torch.LongTensor(np.array(unique_dataset['did'].tolist())).cuda()
            
        else:
            self.posi_id = torch.LongTensor(posi_id).cpu()
            self.neg_id = torch.LongTensor(neg_id).cpu()
            self.rel_diff = torch.Tensor(rel_diff).cpu()
            self.posi_sel = torch.Tensor(posi_sel).cpu()
            self.neg_sel = torch.Tensor(neg_sel).cpu()
            self.tag = torch.LongTensor(tag).cuda()
            unique_dataset = dataset[['did','feature']].drop_duplicates('did')
            if sparse_tag:
                # self.fes = self._get_sparse_feature(unique_dataset['feature'].tolist()).cpu()
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cpu() 
            else:
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cpu()                
            self.didx = torch.LongTensor(np.array(unique_dataset['did'].tolist())).cpu()
        self.use_cuda = use_cuda
        self.sparse_tag = sparse_tag

    def _get_sparse_feature_forOnerow(self, row):
        idx = np.array(row, dtype=int)[:,0] - 1      
        val = np.array(row, dtype=float)[:,1]
        if 699 not in idx:
            idx = np.append(idx, 699) 
            val = np.append(val, 0.)
        sparse_fe = torch.sparse.FloatTensor(torch.LongTensor([idx]), torch.FloatTensor(val))
        return sparse_fe
    
    def _get_sparse_feature(self,dat):
        row_idx_ls = []
        col_idx_ls = []
        val_ls = []
        # dat = dat.apply(pre_trans_row)
        for i in range(len(dat)):
            row_idx_ls.extend([i]*len(dat[i])) 
            col_idx_ls.extend(np.array(dat[i], dtype=int)[:,0] - 1)
            val_ls.extend(np.array(dat[i], dtype=float)[:,1])
        sparse_ts = torch.sparse.FloatTensor(torch.LongTensor([row_idx_ls, col_idx_ls]), torch.FloatTensor(val_ls)).to_dense()
        return sparse_ts
        
    def __getitem__(self, index):
        
        posi_feature = self.fes[torch.nonzero(self.didx==self.posi_id[index])[0]]
        neg_feature = self.fes[torch.nonzero(self.didx==self.neg_id[index])[0]]
        # print(torch.nonzero(self.didx==self.posi_id[index]))
        # print(self.posi_id[index])
        return posi_feature, neg_feature, self.rel_diff[index], self.posi_sel[index], self.neg_sel[index], self.tag[index]

    def __len__(self):
        return len(self.posi_id) 


class Wrap_Dataset_tobit(Dataset):
    """Wrapper, convert <posi_feature, neg_feature, rel_dff> Tensor into Pytorch Dataset"""
    def __init__(self, did, sel_tag, ips_weight, isclick, dataset, use_cuda=True, sparse_tag=False):
        if use_cuda:
            self.did = torch.LongTensor(did).cuda()
            self.sel_tag = torch.Tensor(sel_tag).cuda()
            self.ips_weight = torch.Tensor(ips_weight).cuda()
            self.isclick = torch.Tensor(isclick).cuda()
            unique_dataset = dataset[['did','feature']].drop_duplicates('did')
            if sparse_tag:
                # self.fes = self._get_sparse_feature(unique_dataset['feature'].tolist()).cuda()
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cuda()
            else:
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cuda()               
            self.didx = torch.LongTensor(np.array(unique_dataset['did'].tolist())).cuda()
        else:
            self.did = torch.LongTensor(did).cpu()
            self.sel_tag = torch.Tensor(sel_tag).cpu()
            self.ips_weight = torch.Tensor(ips_weight).cpu()
            self.isclick = torch.Tensor(isclick).cpu()
            unique_dataset = dataset[['did','feature']].drop_duplicates('did')
            if sparse_tag:
                # self.fes = self._get_sparse_feature(unique_dataset['feature'].tolist()).cpu()
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cpu()
            else:
                self.fes = torch.Tensor(np.array(unique_dataset['feature'].tolist())).cpu()              
            self.didx = torch.LongTensor(np.array(unique_dataset['did'].tolist())).cpu()
        self.use_cuda = use_cuda
        self.sparse_tag = sparse_tag

    def __getitem__(self, index):
        feature = self.fes[torch.nonzero(self.didx==self.did[index])[0]]
        # print(torch.nonzero(self.didx==self.posi_id[index]))
        # print(self.posi_id[index])
        return feature, self.isclick[index], self.sel_tag[index], self.ips_weight[index]

    def __len__(self):
        return len(self.did) 

class Wrap_Dataset_tobit2(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, sel_tag, ips_weight, isclick, feature, use_cuda=True, sparse_tag=False):
        if use_cuda:
            self.sel_tag = torch.Tensor(sel_tag).cuda()
            self.ips_weight = torch.Tensor(ips_weight).cuda()
            self.isclick = torch.Tensor(isclick).cuda()
            self.fes = torch.Tensor(feature).cuda()
        else:
            self.sel_tag = torch.Tensor(sel_tag).cuda()
            self.ips_weight = torch.Tensor(ips_weight).cuda()
            self.isclick = torch.Tensor(isclick).cuda()
            self.fes = torch.Tensor(feature).cuda()


    def __getitem__(self, index):
        return self.fes[index], self.isclick[index], self.sel_tag[index], self.ips_weight[index]

    def __len__(self):
        return len(self.sel_tag) 


class Wrap_Dataset_Pairwise(Dataset):
    """Wrapper, convert <posi_feature, neg_feature, rel_dff> Tensor into Pytorch Dataset"""
    def __init__(self, posi_feature, neg_feature, rel_diff, use_cuda=True):
        if use_cuda:
            self.posi_feature = posi_feature.cuda()
            self.neg_feature = neg_feature.cuda()
            self.rel_diff = rel_diff.cuda()
        else:
            self.posi_feature = posi_feature.cpu()
            self.neg_feature = neg_feature.cpu()
            self.rel_diff = rel_diff.cpu()

    def __getitem__(self, index):
        return self.posi_feature[index], self.neg_feature[index], self.rel_diff[index]

    def __len__(self):
        return self.posi_feature.size(0)  


class Wrap_Dataset_Pairwise_DR(Dataset):
    """Wrapper, convert <posi_feature, neg_feature, rel_dff> Tensor into Pytorch Dataset"""
    def __init__(self, posi_feature, neg_feature, rel_diff, dr_weight, use_cuda=True):
        if use_cuda:
            self.posi_feature = posi_feature.cuda()
            self.neg_feature = neg_feature.cuda()
            self.rel_diff = rel_diff.cuda()
            self.dr_weight = dr_weight.cuda()
        else:
            self.posi_feature = posi_feature.cpu()
            self.neg_feature = neg_feature.cpu()
            self.rel_diff = rel_diff.cpu()
            self.dr_weight = dr_weight.cpu()

    def __getitem__(self, index):
        return self.posi_feature[index], self.neg_feature[index], self.rel_diff[index], self.dr_weight[index]

    def __len__(self):
        return self.posi_feature.size(0)