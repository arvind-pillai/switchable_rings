from __future__ import print_function
import json, time, os, sys, glob
import shutil

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import random
from opt_einsum import contract

import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def featurize(batch, device, shuffle_fraction=0.):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S = np.zeros([B, L_max], dtype=np.int32)
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(100))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        #random.shuffle(masked_chains)
        #random.shuffle(visible_chains)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            elif letter in masked_chains: 
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)

        #all_sequence = ''.join([a if a!='-' else 'A' for a in all_sequence])  #replace gaps with Alanine, these parts will be masked
        #all_sequence = ''.join([a if a!='X' else 'A' for a in all_sequence])  #replace gaps with Alanine, these parts will be masked

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices


    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    jumps = ((residue_idx[:,1:]-residue_idx[:,:-1])==1).astype(np.float32)
    phi_mask = np.pad(jumps, [[0,0],[1,0]])
    psi_mask = np.pad(jumps, [[0,0],[0,1]])
    omega_mask = np.pad(jumps, [[0,0],[0,1]])
    dihedral_mask = np.concatenate([phi_mask[:,:,None], psi_mask[:,:,None], omega_mask[:,:,None]], -1) #[B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, dihedral_mask, chain_encoding_all


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed 
    return loss, loss_av

class StructureDataset():
    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                name = entry['name']

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        self.data.append(entry)
                    else:
                        discard_count['too_long'] += 1
                else:
                    print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch
            
batch_size = 7000
max_length = 7000

dataset_train_vol0 = StructureDataset('/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_train_vol0.jsonl', truncate=None, max_length=max_length)
dataset_train_vol1 = StructureDataset('/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_train_vol1.jsonl', truncate=None, max_length=max_length)
dataset_train_vol2 = StructureDataset('/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_train_vol2.jsonl', truncate=None, max_length=max_length)
dataset_train_vol3 = StructureDataset('/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_train_vol3.jsonl', truncate=None, max_length=max_length)
dataset_train_vol4 = StructureDataset('/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_train_vol4.jsonl', truncate=None, max_length=max_length)


dataset_valid = StructureDataset("/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_valid.jsonl", truncate=None, max_length=max_length)

loader_train_vol0 = StructureLoader(dataset_train_vol0, batch_size=batch_size)
loader_train_vol1 = StructureLoader(dataset_train_vol1, batch_size=batch_size)
loader_train_vol2 = StructureLoader(dataset_train_vol2, batch_size=batch_size)
loader_train_vol3 = StructureLoader(dataset_train_vol3, batch_size=batch_size)
loader_train_vol4 = StructureLoader(dataset_train_vol4, batch_size=batch_size)
loader_valid = StructureLoader(dataset_valid, batch_size=batch_size)


# Build basepath for experiment
base_folder = time.strftime('/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p10/', time.localtime())
# base_folder = './log/20Jun10_1039PM/'
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
subfolders = ['checkpoints']
for subfolder in subfolders:
    if not os.path.exists(base_folder + subfolder):
        os.makedirs(base_folder + subfolder)
        
    
# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class TransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = NeighborAttention(num_hidden, num_in, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # Self-attention
        dh = self.attention(h_V, h_E, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

    def step(self, t, h_V, h_E, mask_V=None, mask_attend=None):
        """ Sequential computation of step t of a transformer layer """
        # Self-attention
        h_V_t = h_V[:,t,:]
        dh_t = self.attention.step(t, h_V, h_E, mask_attend)
        h_V_t = self.norm[0](h_V_t + self.dropout(dh_t))

        # Position-wise feedforward
        dh_t = self.dense(h_V_t)
        h_V_t = self.norm[1](h_V_t + self.dropout(dh_t))

        if mask_V is not None:
            mask_V_t = mask_V[:,t].unsqueeze(-1)
            h_V_t = mask_V_t * h_V_t
        return h_V_t

class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden + num_in)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_EV = self.norm1(h_EV)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = h_V + self.dropout(dh)

        # Position-wise feedforward
        dh = self.norm2(h_V)
        dh = self.dense(dh)
        h_V = h_V + self.dropout(dh)

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)
        return

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(device))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, h_V, h_E, mask_attend=None):
        """ Self-attention, graph-structured O(Nk)
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_hidden]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V:            Node update
        """

        # Queries, Keys, Values
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_batch, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2,-1)
        attend_logits = attend_logits / np.sqrt(d)
        
        if mask_attend is not None:
            # Masked softmax
            mask = mask_attend.unsqueeze(2).expand(-1,-1,n_heads,-1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(2,3))
        h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update

    def step(self, t, h_V, h_E, E_idx, mask_attend=None):
        """ Self-attention for a specific time step t
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_in]
            E_idx:          Neighbor indices        [N_batch, N_nodes, K]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V_t:            Node update
        """
        # Dimensions
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads
        d = self.num_hidden / n_heads

        # Per time-step tensors
        h_V_t = h_V[:,t,:]
        h_E_t = h_E[:,t,:,:]
        E_idx_t = E_idx[:,t,:]

        # Single time-step
        h_V_neighbors_t = gather_nodes_t(h_V, E_idx_t)
        E_t = torch.cat([h_E_t, h_V_neighbors_t], -1)

        # Queries, Keys, Values
        Q = self.W_Q(h_V_t).view([n_batch, 1, n_heads, 1, d])
        K = self.W_K(E_t).view([n_batch, n_neighbors, n_heads, d, 1])
        V = self.W_V(E_t).view([n_batch, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_neighbors, n_heads]).transpose(-2,-1)
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            # Masked softmax
            # [N_batch, K] -=> [N_batch, N_heads, K]
            mask_t = mask_attend[:,t,:].unsqueeze(1).expand(-1,n_heads,-1)
            attend = self._masked_softmax(attend_logits, mask_t)
        else:
            attend = F.softmax(attend_logits / np.sqrt(d), -1)

        # Attentive reduction
        h_V_t_update = torch.matmul(attend.unsqueeze(-2), V.transpose(1,2))
        return h_V_t_update

    
class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1, num_embeddings)

    def forward(self, offset):
        #offset: i-j
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature).long()
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1)
        E = self.linear(d_onehot.float())
        return E

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, features_type='full', augment_eps=0., dropout=0.1, num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Feature types
        self.features_type = features_type
        self.feature_dimensions = {
            'coarse': (3, num_positional_embeddings + num_rbf + 7+ num_chain_embeddings),
            'full': (6, num_positional_embeddings + num_rbf*15 + 7*11 + num_chain_embeddings),
            'dist': (6, num_positional_embeddings + num_rbf),
            'hbonds': (3, 2 * num_positional_embeddings),
        }

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.dropout = nn.Dropout(dropout)
        self.linear_chains = nn.Linear(2, num_chain_embeddings)
        # Normalization and embedding
        node_in, edge_in = self.feature_dimensions[features_type]
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, top_k_sample=True, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        if top_k_sample:
            sampled_top_k = np.random.randint(32,self.top_k+1)
        else:
            sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

        # Debug plot KNN
        # print(E_idx[:10,:10])
        # D_simple = mask_2D * torch.zeros(D.size()).scatter(-1, E_idx, torch.ones_like(knn_D))
        # print(D_simple)
        # fig = plt.figure(figsize=(4,4))
        # ax = fig.add_subplot(111)
        # D_simple = D.data.numpy()[0,:,:]
        # plt.imshow(D_simple, aspect='equal')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('D_knn.pdf')
        # exit(0)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        # for i in range(D_count):
        #     fig = plt.figure(figsize=(4,4))
        #     ax = fig.add_subplot(111)
        #     rbf_i = RBF.data.numpy()[0,i,:,:]
        #     # rbf_i = D.data.numpy()[0,0,:,:]
        #     plt.imshow(rbf_i, aspect='equal')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig('rbf{}.pdf'.format(i))
        #     print(np.min(rbf_i), np.max(rbf_i), np.mean(rbf_i))
        # exit(0)
        return RBF

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        # Axis of rotation
        # Replace bad rotation matrices with identity
        # I = torch.eye(3).view((1,1,1,3,3))
        # I = I.expand(*(list(R.shape[:3]) + [-1,-1]))
        # det = (
        #     R[:,:,:,0,0] * (R[:,:,:,1,1] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,1])
        #     - R[:,:,:,0,1] * (R[:,:,:,1,0] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,0])
        #     + R[:,:,:,0,2] * (R[:,:,:,1,0] * R[:,:,:,2,1] - R[:,:,:,1,1] * R[:,:,:,2,0])
        # )
        # det_mask = torch.abs(det.unsqueeze(-1).unsqueeze(-1))
        # R = det_mask * R + (1 - det_mask) * I

        # DEBUG
        # https://math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        # Columns of this are in rotation plane
        # A = R - I
        # v1, v2 = A[:,:,:,:,0], A[:,:,:,:,1]
        # axis = F.normalize(torch.cross(v1, v2), dim=-1)
        return Q

    def _contacts(self, D_neighbors, E_idx, mask_neighbors, cutoff=8):
        """ Contacts """
        D_neighbors = D_neighbors.unsqueeze(-1)
        neighbor_C = mask_neighbors * (D_neighbors < cutoff).type(torch.float32)
        return neighbor_C

    def _hbonds(self, X, E_idx, mask_neighbors, eps=1E-3):
        """ Hydrogen bonds and contact map
        """
        X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

        # Virtual hydrogens
        X_atoms['C_prev'] = F.pad(X_atoms['C'][:,1:,:], (0,0,0,1), 'constant', 0)
        X_atoms['H'] = X_atoms['N'] + F.normalize(
             F.normalize(X_atoms['N'] - X_atoms['C_prev'], -1)
          +  F.normalize(X_atoms['N'] - X_atoms['CA'], -1)
        , -1)

        def _distance(X_a, X_b):
            return torch.norm(X_a[:,None,:,:] - X_b[:,:,None,:], dim=-1)

        def _inv_distance(X_a, X_b):
            return 1. / (_distance(X_a, X_b) + eps)

        # DSSP vacuum electrostatics model
        U = (0.084 * 332) * (
              _inv_distance(X_atoms['O'], X_atoms['N'])
            + _inv_distance(X_atoms['C'], X_atoms['H'])
            - _inv_distance(X_atoms['O'], X_atoms['H'])
            - _inv_distance(X_atoms['C'], X_atoms['N'])
        )

        HB = (U < -0.5).type(torch.float32)
        neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1),  E_idx)
        # print(HB)
        # HB = F.sigmoid(U)
        # U_np = U.cpu().data.numpy()
        # # plt.matshow(np.mean(U_np < -0.5, axis=0))
        # plt.matshow(HB[0,:,:])
        # plt.colorbar()
        # plt.show()
        # D_CA = _distance(X_atoms['CA'], X_atoms['CA'])
        # D_CA = D_CA.cpu().data.numpy()
        # plt.matshow(D_CA[0,:,:] < contact_D)
        # # plt.colorbar()
        # plt.show()
        # exit(0)
        return neighbor_HB

    def _get_orientations(self, A, B, C, E_idx, eps=1e-6):
        v1 = A-B #[B, L, 3]
        v2 = C-B #[B, L, 3]
        e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps) #[B, L, 3]
        u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[:,:,None]*e1) #[B, L, 3]
        e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
        e3 = torch.cross(e1, e2, dim=-1)
        O = torch.cat([e1[:,:,:,None], e2[:,:,:,None], e3[:,:,:,None]], axis=-1) #[L,3,3] - rotation matrix
        O = O.view(list(O.shape[:2]) + [9])

        O_neighbors = gather_nodes(O, E_idx)
        B_neighbors = gather_nodes(B, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3,3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

        # Rotate into local reference frames
        dX = B_neighbors - B.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1)
        return O_features

    def _orientations_coarse(self, X, N, C, Cb, atom_O, E_idx, eps=1e-6):
        # Pair features

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1+eps, 1-eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0,0,1,2), 'constant', 0)
        O_list = []
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)
        O_list.append(O)
        for O in O_list:
            O_neighbors = gather_nodes(O, E_idx)
            X_neighbors = gather_nodes(X, E_idx)
            
            # Re-view as rotation matrices
            O = O.view(list(O.shape[:2]) + [3,3])
            O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

            # Rotate into local reference frames
            dX = X_neighbors - X.unsqueeze(-2)
            dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
            dU = F.normalize(dU, dim=-1)
            R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
            Q = self._quaternions(R)

            # Orientation features
            O_features = torch.cat((dU,Q), dim=-1)

        return AD_features, O_features

    def _dihedrals(self, X, dihedral_mask, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)

        # print(cosD.cpu().data.numpy().flatten())
        # print(omega.sum().cpu().data.numpy().flatten())

        # Bond angle calculation
        # A = torch.acos(-(u_1 * u_0).sum(-1))

        # DEBUG: Ramachandran plot
        # x = phi.cpu().data.numpy().flatten()
        # y = psi.cpu().data.numpy().flatten()
        # plt.scatter(x * 180 / np.pi, y * 180 / np.pi, s=1, marker='.')
        # plt.xlabel('phi')
        # plt.ylabel('psi')
        # plt.axis('square')
        # plt.grid()
        # plt.axis([-180,180,-180,180])
        # plt.show()

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D)*dihedral_mask, torch.sin(D)*dihedral_mask), 2)
        return D_features

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B


    def forward(self, X, L, mask, residue_idx, dihedral_mask, chain_labels, top_k_sample=False):
        """ Featurize coordinates as an attributed graph """
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        # Build k-Nearest Neighbors graph
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask, top_k_sample)

        # Pairwise features
        AD_features, O_features = self._orientations_coarse(X_ca, X[:,:,0,:], X[:,:,2,:], Cb, X[:,:,3,:], E_idx)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(X[:,:,0,:], X[:,:,0,:], E_idx)) #N-N
        RBF_all.append(self._get_rbf(X[:,:,2,:], X[:,:,2,:], E_idx)) #C-C
        RBF_all.append(self._get_rbf(X[:,:,3,:], X[:,:,3,:], E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(X[:,:,1,:], X[:,:,0,:], E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(X[:,:,1,:], X[:,:,2,:], E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(X[:,:,1,:], X[:,:,3,:], E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(X[:,:,1,:], Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(X[:,:,0,:], X[:,:,2,:], E_idx)) #N-C
        RBF_all.append(self._get_rbf(X[:,:,0,:], X[:,:,3,:], E_idx)) #N-O
        RBF_all.append(self._get_rbf(X[:,:,0,:], Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, X[:,:,2,:], E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, X[:,:,3,:], E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(X[:,:,3,:], X[:,:,2,:], E_idx)) #O-C
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        N = X[:,:,0,:]
        Ca = X[:,:,1,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
  
        O_features_list = []
        O_features_list.append(O_features)
        O_features_list.append(self._get_orientations(Cb, N, Ca, E_idx))
        O_features_list.append(self._get_orientations(Ca, Cb, C, E_idx))
        O_features_list.append(self._get_orientations(Cb, Ca, O, E_idx))
        O_features_list.append(self._get_orientations(N, Ca, C, E_idx))
        O_features_list.append(self._get_orientations(Ca, N, O, E_idx))
        O_features_list.append(self._get_orientations(O, C, Ca, E_idx))
        O_features_list.append(self._get_orientations(C, Cb, N, E_idx))
        O_features_list.append(self._get_orientations(N, O, Cb, E_idx))
        O_features_list.append(self._get_orientations(C, O, Cb, E_idx))
        O_features_list.append(self._get_orientations(O, C, N, E_idx))
        O_features_all = torch.cat(tuple(O_features_list), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]
        E_positional = self.embeddings(offset)

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_chains = torch.nn.functional.one_hot(E_chains, 2).float()
        E_chains = self.linear_chains(E_chains)

        if self.features_type == 'coarse':
            # Coarse backbone features
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'hbonds':
            # Hydrogen bonds and contacts
            neighbor_HB = self._hbonds(X, E_idx, mask_neighbors)
            neighbor_C = self._contacts(D_neighbors, E_idx, mask_neighbors)
            # Dropout
            neighbor_C = self.dropout(neighbor_C)
            neighbor_HB = self.dropout(neighbor_HB)
            # Pack
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1,-1,-1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1,-1,-1, int(self.num_positional_embeddings / 2))
            E = torch.cat((E_positional, neighbor_C, neighbor_HB), -1)
        elif self.features_type == 'full':
            # Full backbone angles
            V = self._dihedrals(X, dihedral_mask)
            E = torch.cat((E_chains, E_positional, RBF_all, O_features_all), -1)
        elif self.features_type == 'dist':
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((E_positional, RBF), -1)

        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        # DEBUG
        # U = (np.nan * torch.zeros(X.size(0),X.size(1),X.size(1),3)).scatter(2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), E[:,:,:,:3])
        # plt.imshow(U.data.numpy()[0,:,:,0])
        # plt.show()
        # exit(0)
        return V, E, E_idx     

    
class Struct2Seq(nn.Module):
    def __init__(self, num_letters, node_features, edge_features,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=64, protein_features='full', augment_eps=0.05,
        dropout=0.1, forward_attention_decoder=True, use_mpnn=False):
        """ Graph labeling network """
        super(Struct2Seq, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors,
            features_type=protein_features, augment_eps=augment_eps,
            dropout=dropout
        )

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        layer = TransformerLayer if not use_mpnn else MPNNLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.forward_attention_decoder = forward_attention_decoder
        self.decoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _autoregressive_mask(self, E_idx, order_mask=None):
        if order_mask!=None:
            mask = torch.gather(order_mask, 2, E_idx)
        else:
            N_nodes = E_idx.size(1)
            ii = torch.arange(N_nodes).to(device)
            ii = ii.view((1, -1, 1))
            mask = E_idx - ii < 0
            mask = mask.type(torch.float32)

            # Debug 
            # mask_scatter = torch.zeros(E_idx.shape[0],E_idx.shape[1],E_idx.shape[1]).scatter(-1, E_idx, mask)
            # mask_reduce = gather_edges(mask_scatter.unsqueeze(-1), E_idx).squeeze(-1)
            # plt.imshow(mask_reduce.data.numpy()[0,:,:])
            # plt.show()
            # plt.imshow(mask.data.numpy()[0,:,:])
            # plt.show()
        return mask

    def forward(self, X, S, L, mask, chain_M, residue_idx, dihedral_mask, chain_encoding_all, top_k_sample=False):
        """ Graph-conditioned sequence model """

        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask, residue_idx, dihedral_mask, chain_encoding_all, top_k_sample)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_EV, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_ES_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx)

        # Decoder uses masked self-attention
        decoding_order = torch.argsort(torch.argsort(chain_M*(torch.abs(torch.randn(chain_M.shape))+1e-3).to(device)))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.eye(mask_size)[decoding_order].to(device) #[B, L, L]
        order_mask_backward = contract('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size))).to(device), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = self._autoregressive_mask(E_idx, order_mask_backward).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend

        
        if self.forward_attention_decoder:
            mask_fw = mask_1D * (1. - mask_attend)
            h_ESV_encoder_fw = mask_fw * h_ESV_encoder
        else:
            h_ESV_encoder_fw = 0
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        logits = self.W_out(h_V) 
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

NUM_H=192
model = Struct2Seq(num_letters=21, node_features=NUM_H, edge_features=NUM_H, hidden_dim=NUM_H, use_mpnn=True, num_encoder_layers=3, num_decoder_layers=3, k_neighbors=64, dropout=0.3, augment_eps=0.1)
model.to(device)
print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )

#PATH = '/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p10/checkpoints/epoch27_step135000.pt'
total_step = 0
epoch = 0
#checkpoint = torch.load(PATH)
#model.load_state_dict(checkpoint['model_state_dict'])
optimizer = get_std_opt(model.parameters(), NUM_H, total_step)
#optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(base_folder)

start_train = time.time()
epoch_losses_train, epoch_losses_valid = [], []
epoch_checkpoints = []

def get_interface_weights(N, Ca, C, mask, mask_self, top_k=5, eps=1e-6):
    "N, Ca, C - [B, L, 3], mask - [B, L], chain_Ls_list_list - [[64, 78, 40], [[65, 124], ...]"
    b = Ca - N 
    c = C - Ca 
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
    dX = torch.unsqueeze(Cb,1) - torch.unsqueeze(Cb,2)
    D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
    D_max = 1000.0
    D_adjust = D + D_max * (1. - mask_2D) + D_max * (1. - mask_self)
    interface_contacts_mask = ((D_adjust < 8.0).sum(-1) > 0).float()
    interface_weights = 1.0+0.0*interface_contacts_mask
    return interface_weights, interface_contacts_mask



# Log files
logfile = base_folder + 'log.txt'
with open(logfile, 'w') as f:
    f.write('Epoch\tTrain\tValidation\n')


train_loader_list = [loader_train_vol0, loader_train_vol1, loader_train_vol2, loader_train_vol3, loader_train_vol4]
#train_loader_list = [loader_train_vol0]

for e in range(300):
    e = epoch + e

    model.train()
    train_sum, train_weights = 0., 0.
    train_sum_ori, train_weights_ori = 0., 0.
    train_sum_interface, train_weights_interface = 0., 0.
    train_sum_other, train_weights_other = 0., 0.
    for loader_ in train_loader_list:
        for train_i, batch in enumerate(loader_):
            if train_i < 1000: 
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, dihedral_mask, chain_encoding_all = featurize(batch, device) #shuffle fraction can be specified
                elapsed_featurize = time.time() - start_batch
                interface_weights, interface_contacts_mask = get_interface_weights(X[:,:,0,:], X[:,:,1,:], X[:,:,2,:], mask, mask_self, top_k=5, eps=1e-6)
                optimizer.zero_grad()
                log_probs = model(X, S, lengths, mask, chain_M, residue_idx, dihedral_mask, chain_encoding_all, top_k_sample=True)
                mask_for_loss = mask*chain_M*interface_weights
                mask_for_loss_ori = mask*chain_M
                _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss) #weight smoothinb  can be specified
                loss_av_smoothed.backward()
                optimizer.step()
    
                loss, loss_av = loss_nll(S, log_probs, mask_for_loss)
                loss_ori, loss_av_ = loss_nll(S, log_probs, mask_for_loss_ori)
                mask_for_loss_interface = mask*chain_M*interface_contacts_mask
                mask_for_loss_other = mask*chain_M*(1.0-interface_contacts_mask)
                loss_interface, loss_av_ = loss_nll(S, log_probs, mask_for_loss_interface)
                loss_other, loss_av_ = loss_nll(S, log_probs, mask_for_loss_other)
    
                # Timing
                elapsed_batch = time.time() - start_batch
                elapsed_train = time.time() - start_train
                total_step += 1
    
                # Accumulate true loss
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
    
                train_sum_ori += torch.sum(loss_ori * mask_for_loss_ori).cpu().data.numpy()
                train_weights_ori += torch.sum(mask_for_loss_ori).cpu().data.numpy()
    
                train_sum_interface += torch.sum(loss_interface * mask_for_loss_interface).cpu().data.numpy()
                train_weights_interface += torch.sum(mask_for_loss_interface).cpu().data.numpy()
    
                train_sum_other += torch.sum(loss_other * mask_for_loss_other).cpu().data.numpy()
                train_weights_other += torch.sum(mask_for_loss_other).cpu().data.numpy()
    
   
    model.eval()
    with torch.no_grad():
        validation_sum, validation_weights = 0., 0.
        validation_sum_ori, validation_weights_ori = 0., 0.
        validation_sum_interface, validation_weights_interface = 0., 0.
        validation_sum_other, validation_weights_other = 0., 0.
        for _, batch in enumerate(loader_valid):
            X, S, mask, lengths, chain_M, residue_idx, mask_self, dihedral_mask, chain_encoding_all = featurize(batch, device)
            interface_weights, interface_contacts_mask = get_interface_weights(X[:,:,0,:], X[:,:,1,:], X[:,:,2,:], mask, mask_self, top_k=5, eps=1e-6)
            log_probs = model(X, S, lengths, mask, chain_M, residue_idx, dihedral_mask, chain_encoding_all, top_k_sample=False)
            mask_for_loss = mask*chain_M*interface_weights
            mask_for_loss_ori = mask*chain_M
            loss, loss_av = loss_nll(S, log_probs, mask_for_loss)
            loss_ori, loss_av_ = loss_nll(S, log_probs, mask_for_loss_ori)
            mask_for_loss_interface = mask*chain_M*interface_contacts_mask
            mask_for_loss_other = mask*chain_M*(1.0-interface_contacts_mask)
            loss_interface, loss_av_ = loss_nll(S, log_probs, mask_for_loss_interface)
            loss_other, loss_av_ = loss_nll(S, log_probs, mask_for_loss_other)
            # Accumulate
            validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            validation_sum_ori += torch.sum(loss_ori * mask_for_loss_ori).cpu().data.numpy()
            validation_weights_ori += torch.sum(mask_for_loss_ori).cpu().data.numpy()

            validation_sum_interface += torch.sum(loss_interface * mask_for_loss_interface).cpu().data.numpy()
            validation_weights_interface += torch.sum(mask_for_loss_interface).cpu().data.numpy()

            validation_sum_other += torch.sum(loss_other * mask_for_loss_other).cpu().data.numpy()
            validation_weights_other += torch.sum(mask_for_loss_other).cpu().data.numpy()

    train_loss = train_sum / train_weights
    train_loss_ori = train_sum_ori / train_weights_ori
    train_loss_interface = train_sum_interface / train_weights_interface
    train_loss_other = train_sum_other / train_weights_other
    train_perplexity = np.exp(train_loss)
    train_perplexity_ori = np.exp(train_loss_ori)
    train_perplexity_interface = np.exp(train_loss_interface)
    train_perplexity_other = np.exp(train_loss_other)
    validation_loss = validation_sum / validation_weights
    validation_loss_ori = validation_sum_ori / validation_weights_ori
    validation_loss_interface = validation_sum_interface / validation_weights_interface
    validation_loss_other = validation_sum_other / validation_weights_other
    validation_perplexity = np.exp(validation_loss)
    validation_perplexity_ori = np.exp(validation_loss_ori)
    validation_perplexity_interface = np.exp(validation_loss_interface)
    validation_perplexity_other = np.exp(validation_loss_other)

    train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)
    train_perplexity_ori_ = np.format_float_positional(np.float32(train_perplexity_ori), unique=False, precision=3)
    train_perplexity_interface_ = np.format_float_positional(np.float32(train_perplexity_interface), unique=False, precision=3)
    train_perplexity_other_ = np.format_float_positional(np.float32(train_perplexity_other), unique=False, precision=3)

    validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
    validation_perplexity_ori_ = np.format_float_positional(np.float32(validation_perplexity_ori), unique=False, precision=3)
    validation_perplexity_interface_ = np.format_float_positional(np.float32(validation_perplexity_interface), unique=False, precision=3)
    validation_perplexity_other_ = np.format_float_positional(np.float32(validation_perplexity_other), unique=False, precision=3)
 
    with open(logfile, 'a') as f:
        f.write(f'epoch: {e}, step: {total_step}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_ori: {train_perplexity_ori_}, valid_ori: {validation_perplexity_ori_}, train_interface: {train_perplexity_interface_}, valid_interface: {validation_perplexity_interface_}, train_other: {train_perplexity_other_}, valid_other: {validation_perplexity_other_}\n')

    # Save the model
    checkpoint_filename = base_folder+'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step)
    torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            }, checkpoint_filename)

    epoch_losses_valid.append(validation_perplexity)
    epoch_losses_train.append(train_perplexity)
    epoch_checkpoints.append(checkpoint_filename)
