import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch

from sklearn.cluster import KMeans
import os
import os.path as osp
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

class Base:

    def __init__(self, adj, features, device):
        self.adj = adj
        self.features = features.to(device)
        self.device = device
        self.cached_adj_norm = None

    def get_adj_norm(self):
        if self.cached_adj_norm is None:
            adj_norm = preprocess_adj(self.adj, self.device)
            self.cached_adj_norm= adj_norm
        return self.cached_adj_norm

    def make_loss(self, embeddings):
        return 0

    def transform_data(self):
        return self.get_adj_norm(), self.features

class PairwiseAttrSim(Base):

    def __init__(self, nhid, args,device,  regression=True):


        self.args = args
        self.device = device

        self.regression = regression
        self.nclass = 1
        if regression:
            self.linear = nn.Linear(nhid, self.nclass).to(device)
        else:
            self.linear = nn.Linear(nhid, 2).to(device)

    def transform_data(self):
        return self.get_adj_norm(), self.features

    def make_loss(self, skill,embeddings):
        if self.regression:
            return self.regression_loss(skill,embeddings)


    def regression_loss(self,skill, embeddings):
        k = 10000
        if len(skill[0]) > k:
            sampled = np.random.choice(len(skill[0]), k, replace=False)

            embeddings0 = embeddings[skill[0][sampled]]
            embeddings1 = embeddings[skill[1][sampled]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))

            loss = F.mse_loss(embeddings, torch.zeros_like(embeddings), reduction='mean')
        else:
            embeddings0 = embeddings[skill[0]]
            embeddings1 = embeddings[skill[1]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, torch.zeros_like(embeddings), reduction='mean')
        # print(loss)
        return loss


class MemoryBankModule(torch.nn.Module):
    """Memory bank implementation

    This is a parent class to all loss functions implemented by the lightly
    Python package. This way, any loss can be used with a memory bank if
    desired.

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.

    Examples:
        >>> class MyLossFunction(MemoryBankModule):
        >>>
        >>>     def __init__(self, memory_bank_size: int = 2 ** 16):
        >>>         super(MyLossFunction, self).__init__(memory_bank_size)
        >>>
        >>>     def forward(self, output: torch.Tensor,
        >>>                 labels: torch.Tensor = None):
        >>>
        >>>         output, negatives = super(
        >>>             MyLossFunction, self).forward(output)
        >>>
        >>>         if negatives is not None:
        >>>             # evaluate loss with negative samples
        >>>         else:
        >>>             # evaluate loss without negative samples

    """

    def __init__(self, size: int = 2 ** 16):

        super(MemoryBankModule, self).__init__()

        if size < 0:
            msg = f'Illegal memory bank size {size}, must be non-negative.'
            raise ValueError(msg)

        self.size = size

        self.bank = None
        self.bank_ptr = None

    @torch.no_grad()
    def _init_memory_bank(self, dim: int):
        """Initialize the memory bank if it's empty

        Args:
            dim:
                The dimension of the which are stored in the bank.

        """
        # create memory bank
        # we could use register buffers like in the moco repo
        # https://github.com/facebookresearch/moco but we don't
        # want to pollute our checkpoints
        self.bank = torch.randn(dim, self.size)
        self.bank = torch.nn.functional.normalize(self.bank, dim=0)
        self.bank_ptr = torch.LongTensor([0])

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: torch.Tensor):
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)

        if ptr + batch_size >= self.size:
            self.bank[:, ptr:] = batch[:self.size - ptr].T.detach()
            self.bank_ptr[0] = 0
        else:
            self.bank[:, ptr:ptr + batch_size] = batch.T.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(self,
                output: torch.Tensor,
                labels: torch.Tensor = None,
                update: bool = False):
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            labels:
                Should always be None, will be ignored.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank.

        """

        # no memory bank, return the output
        if self.size == 0:
            return output, None

        _, dim = output.shape

        # initialize the memory bank if it is not already done
        if self.bank is None:
            self._init_memory_bank(dim)

        # query and update memory bank
        bank = self.bank.clone().detach()

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output)

        return output, bank


class NTXentLoss(MemoryBankModule):
    """Implementation of the Contrastive Cross Entropy Loss.

    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like
    the one described in the MoCo[1] paper.

    - [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    - [1] MoCo, 2020, https://arxiv.org/abs/1911.05722

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Number of negative samples to store in the memory bank.
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.
        gather_distributed:
            If True then negatives from all gpus are gathered before the
            loss calculation. This flag has no effect if memory_bank_size > 0.

    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:

        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)

    """

    def __init__(self,device,
                 temperature: float = 0.07,
                 memory_bank_size: int = 0,
                 gather_distributed: bool = False):
        super(NTXentLoss, self).__init__(size=memory_bank_size)
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8
        self.linear = nn.Linear(128, 128).to(device)
        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))

    def get_loss(self,pos_pair, weaker_pos_pair, neg_pair,embeddings,device):
        embeddings = self.linear(embeddings)
        q = torch.tensor([],device=device)
        k = torch.tensor([],device=device)
        l_neg = torch.tensor([],device =device)
        l_neg_weaker = torch.tensor([],device=device)
        q_weaker = torch.tensor([],device=device)
        k_weaker = torch.tensor([],device=device)
        number = 0
        for i in pos_pair.keys():
            q = torch.cat([q,embeddings[[i]]],dim=0)
            index = np.random.choice(pos_pair[i],1)
            k = torch.cat([k,embeddings[index]])
            # negtive skill
            neg_indexs = np.array([x for x in range(embeddings.size(0))])

            mask = np.ones(embeddings.size(0))
            for j in neg_pair[i]:
                mask[j] = 0
            mask = mask > 0

            neg_indexs = neg_indexs[mask]
            neg_index = np.random.choice(neg_indexs,256,replace=False)
            neg = embeddings[neg_index]


            l_neg_part = torch.einsum('nc,ck->nk', [embeddings[i].resize(1,128), neg.T])
            l_neg = torch.cat([l_neg, l_neg_part], dim=0)
            # positive logits: Nx1

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], device=device,dtype=torch.long)

        loss = self.cross_entropy(logits, labels)


        for i in weaker_pos_pair.keys():
            q_weaker = torch.cat([q_weaker,embeddings[[i]]],dim=0)
            index = np.random.choice(weaker_pos_pair[i], 1)
            k_weaker = torch.cat([k_weaker, embeddings[index]])
            neg_indexs = np.array([x for x in range(embeddings.size(0))])
            mask = np.ones(embeddings.size(0))
            for j in neg_pair[i]:
                mask[j] = 0
            mask = mask > 0
            neg_indexs = neg_indexs[mask]
            neg_index = np.random.choice(neg_indexs, 256, replace=False)
            neg = embeddings[neg_index]
            l_neg_part = torch.einsum('nc,ck->nk', [embeddings[[i]], neg.T])
            l_neg_weaker = torch.cat([l_neg_weaker, l_neg_part], dim=0)
        l_weaker_pos = torch.einsum('nc,nc->n', [q_weaker, k_weaker]).unsqueeze(-1)
        # logits: Nx(1+K)
        logits_weaker = torch.cat([l_weaker_pos, l_neg_weaker], dim=1)
        # apply temperature
        labels1 = torch.zeros(logits_weaker.shape[0], device=device, dtype=torch.long)
        logits_weaker /= self.temperature
        # create labels

        loss_weaker = self.cross_entropy(logits_weaker, labels1)
        loss_total = loss + 0.1 * loss_weaker

        return loss_total

    def get_loss1(self,pos_pair, weaker_pos_pair, neg_pair,embeddings,device,fl_train_skill,cl_train_skill):
        index_all =[]
        index_weaker_all = []
        new_pos_pair ={}
        new_weaker_pos_pair = {}
        new_neg_pair = {}

        for i in range(len(fl_train_skill)):
            index = np.array([x for x in range(embeddings.size(0))])
            index = index[fl_train_skill[i].cpu()]

            for j in index:
                if j not in index_all:
                    index_all.append(j)

        for i in index_all:
            if i in weaker_pos_pair.keys():
                new_weaker_pos_pair[i] = weaker_pos_pair[i]

        for i in range(len(cl_train_skill)):
            index_weaker = np.array([x for x in range(embeddings.size(0))])
            index_weaker = index_weaker[cl_train_skill[i].cpu()]
            for j in index_weaker:
                if j not in index_weaker_all:
                    index_weaker_all.append(j)

        for i in index_weaker_all:
            if i in pos_pair.keys():
                new_pos_pair[i] = pos_pair[i]

        return self.get_loss(new_pos_pair, new_weaker_pos_pair, neg_pair,embeddings,device)


def preprocess_features(features, device):
    return features.to(device)

def preprocess_adj(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = aug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def preprocess_adj_noloop(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = noaug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def noaug_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

