from __future__ import print_function
import json, time, os, sys, glob
import shutil
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import os.path

from opt_einsum import contract
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argparser.add_argument("--checkpoint_path", type=str, help="Path to the model checkpoint")
argparser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for the model")
argparser.add_argument("--num_layers", type=int, default=3, help="Number of layers for the model")
argparser.add_argument("--protein_features", type=str, default='full', help="full or coarse")
argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl ")
argparser.add_argument("--chain_id_jsonl", type=str, help="Path to a dictionary with masked and visible chains")
argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
argparser.add_argument("--out_folder", type=str, help="Path to a folder to output processed pdb files /home/out/")
argparser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
argparser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
argparser.add_argument("--max_length", type=int, default=500, help="Max sequence length")
argparser.add_argument("--sampling_temp", type=str, default="0.2", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")
argparser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
argparser.add_argument("--backbone_noise", type=float, default=0.05, help="Standard deviation of Gaussian noise to add to backbone atoms")
argparser.add_argument("--decoding_order", type=str, default='random', help="Decoding order ['forward', 'random']")
argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
argparser.add_argument("--num_connections", type=int, default=30, help="Number of neighbors each residue is connected to, default 30, maximum 64, higher number leads to better interface design but will cost more to run the model.")
argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
args = argparser.parse_args()

from mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq
from mpnn_utils import StructureDataset, StructureLoader, MPNNLayer, PositionWiseFeedForward, ProteinFeatures

folder_for_outputs = args.out_folder

NUM_BATCHES = args.num_seq_per_target//args.batch_size
BATCH_COPIES = args.batch_size
temperatures = [float(item) for item in args.sampling_temp.split()]
omit_AAs_list = args.omit_AAs
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

global DECODING_ORDER
DECODING_ORDER = args.decoding_order

omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def extend(a,b,c, L,A,D):
  '''
  input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
  output: 4th coord
  '''
  N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
  bc = N(b-c)
  n = N(np.cross(b-a, bc))
  m = [bc,np.cross(n,bc),n]
  d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
  return c + sum([m*d for m,d in zip(m,d)])


def featurize(batch, device, chain_dict, fixed_position_dict=None, omit_AA_dict=None):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(100))]
    chain_letters = init_alphabet + extra_alphabet
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    for i, b in enumerate(batch):
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[b['name']] #masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:3]=='seq']
            visible_chains = []
        num_chains = b['num_of_chains']
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)
        mask_dict = {}
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}']], 1) #[chain_lenght,4,3]
                N = x_chain[:,0]
                CA = x_chain[:,1]
                C = x_chain[:,2]
                O = extend(np.roll(N,-1,axis=0),CA,C,1.231,2.108,-3.142)
                x_chain = np.concatenate([N[:,None], CA[:,None], C[:,None], O[:,None]], 1)
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #1.0 for masked
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}']], 1) #[chain_lenght,4,3]
                N = x_chain[:,0]
                CA = x_chain[:,1]
                C = x_chain[:,2]
                O = extend(np.roll(N,-1,axis=0),CA,C,1.231,2.108,-3.142)
                x_chain = np.concatenate([N[:,None], CA[:,None], C[:,None], O[:,None]], 1)
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict!=None:
                    fixed_pos_list = fixed_position_dict[b['name']][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list)-1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict!=None:
                    for item in omit_AA_dict[b['name']][letter]:
                        idx_AA = np.array(item[0])-1
                        AA_idx = np.array([np.argwhere(np.array(list(alphabet))== AA)[0][0] for AA in item[1]]).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:,0], idx_[:,1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
        
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)
        m_pos = np.concatenate(fixed_position_mask_list,0) #[L,], 1.0 for places that need to be predicted

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        m_pos_pad = np.pad(m_pos, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list,0), [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad
        chain_M_pos[i,:] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)


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
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    return X, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask

    
class Struct2Seq(nn.Module):
    def __init__(self, num_letters, node_features, edge_features,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=30, protein_features='full', augment_eps=0.05,
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
        return mask


    def forward(self, X, S, L, mask, chain_encoding_all, chain_M, randn, residue_idx, dihedral_mask):
        """ Graph-conditioned sequence model """

        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask, residue_idx, dihedral_mask, chain_encoding_all, False)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_ES_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx)
        decoding_order = torch.argsort(torch.argsort(chain_M*(torch.abs(randn)+1e-3).to(device)))
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
            h_V = layer(h_V, h_ESV, mask_V=mask)

        logits = self.W_out(h_V) 
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def sample(self, X, L, randn, S_true, chain_mask, chain_encoding_all, residue_idx, dihedral_mask, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None):
        """ Autoregressive decoding of a model """
         # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask, residue_idx, dihedral_mask, chain_encoding_all, False)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)
        
        # Decoder alternates masked self-attention
        chain_M = chain_mask*chain_M_pos
        decoding_order = torch.argsort(torch.argsort(chain_M*(torch.abs(randn)+1e-3).to(device)))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.eye(mask_size)[decoding_order].to(device) #[B, L, L]
        order_mask_backward = contract('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size))).to(device), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = self._autoregressive_mask(E_idx, order_mask_backward).unsqueeze(-1)

        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21))
        h_S = torch.zeros_like(h_V)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64).to(device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=V.device)
        constant_bias = torch.tensor(bias_AAs_np, device=V.device)
        t_order = torch.argmax(permutation_matrix_reverse,axis=-1) 
        chain_mask_combined = chain_mask*chain_M_pos 
        omit_AA_mask_flag = omit_AA_mask != None
        for t_ in range(N_nodes):
            t = t_order[:,t_] #[B]
            chain_mask_gathered = torch.gather(chain_mask_combined, 1, t[:,None]) #[B]
            if (chain_mask_gathered==0).all():
                S_t = torch.gather(S_true, 1, t[:,None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                # Stale relational features for future states
                h_ESV_encoder_t = torch.gather(mask_fw, 1, t[:,None,None,None].repeat(1,1,mask_fw.shape[-2], mask_fw.shape[-1])) * cat_neighbors_nodes(h_V, h_ES_t, E_idx_t)
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_ESV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(
                    h_V_t, h_ESV_t, mask_V=torch.gather(mask, 1, t[:,None])[:,0,]
                ))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature, dim=-1)
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t = torch.multinomial(probs, 1)
            S_true_gathered = torch.gather(S_true, 1, t[:,None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()	
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
            S.scatter_(1, t[:,None], S_t)
        return S
    
    
dataset_valid = StructureDataset(args.jsonl_path, truncate=None, max_length=args.max_length)

if os.path.isfile(args.chain_id_jsonl):
    with open(args.chain_id_jsonl, 'r') as json_file:
        json_list = list(json_file)
    
    for json_str in json_list:
        chain_id_dict = json.loads(json_str)
else:
    chain_id_dict = None
    
if os.path.isfile(args.fixed_positions_jsonl):
    with open(args.fixed_positions_jsonl, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        fixed_positions_dict = json.loads(json_str)
    print('Fixed positions dictionary is loaded')
else:
    print('Fixed positions dictionary is NOT loaded, or NOT provided')
    fixed_positions_dict = None



if os.path.isfile(args.omit_AA_jsonl):
    with open(args.omit_AA_jsonl, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        omit_AA_dict = json.loads(json_str)
    print('Omit AA dictionary is loaded')
else:
    print('Omit AA dictionary is NOT loaded, or NOT provided')
    omit_AA_dict = None


if os.path.isfile(args.bias_AA_jsonl):
    with open(args.bias_AA_jsonl, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        bias_AA_dict = json.loads(json_str)
    print('AA bias dictionary is loaded')
else:
    print('AA bias dictionary is not loaded, or not provided')
    bias_AA_dict = None

bias_AAs_np = np.zeros(len(alphabet))
if bias_AA_dict:
        for n, AA in enumerate(alphabet):
                if AA in list(bias_AA_dict.keys()):
                        bias_AAs_np[n] = bias_AA_dict[AA]


model = Struct2Seq(num_letters=21, node_features=args.hidden_dim, edge_features=args.hidden_dim, hidden_dim=args.hidden_dim, num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers, use_mpnn=True, protein_features=args.protein_features, augment_eps=args.backbone_noise, k_neighbors=args.num_connections)
model.to(device)
checkpoint = torch.load(args.checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Build paths for experiment
base_folder = folder_for_outputs
if base_folder[-1] != '/':
    base_folder = base_folder + '/'
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
for subfolder in ['alignments', 'scores']:
    if not os.path.exists(base_folder + subfolder):
        os.makedirs(base_folder + subfolder)

# Timing
start_time = time.time()
total_residues = 0
protein_list = []
total_step = 0
# Validation epoch
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    print('Generating sequences...')
    for ix, protein in enumerate(dataset_valid):
        score_list = []
        batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask = featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict)
        randn_1 = torch.randn(chain_M.shape).to(device)
        log_probs = model(X, S, lengths, mask, chain_encoding_all, chain_M*chain_M_pos, randn_1, residue_idx, dihedral_mask)
        mask_for_loss = mask*chain_M*chain_M_pos
        scores = _scores(S, log_probs, mask_for_loss)
        native_score = scores.cpu().data.numpy()
        # Generate some sequences
        ali_file = base_folder + '/alignments/' + batch_clones[0]['name'] + '.fa'
        score_file = base_folder + '/scores/' + batch_clones[0]['name'] + '.npy'
        name_ = batch_clones[0]['name']
        print(f'Generating sequences for: {name_}')
        t0 = time.time()
        with open(ali_file, 'w') as f:
            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    randn_2 = torch.randn(chain_M.shape).to(device)
                    S_sample = model.sample(X, lengths, randn_2, S, chain_M, chain_encoding_all, residue_idx, dihedral_mask, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask).to(device)
                    # Compute scores
                    log_probs = model(X, S_sample, lengths, mask, chain_encoding_all, chain_M*chain_M_pos, randn_2, residue_idx, dihedral_mask)
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()
                    for b_ix in range(BATCH_COPIES):
                        masked_chain_length_list = masked_chain_length_list_list[b_ix]
                        masked_list = masked_list_list[b_ix]
                        seq_recovery_rate = torch.sum(torch.sum(torch.eye(21).to(device)[S[b_ix]]*torch.eye(21).to(device)[S_sample[b_ix]],axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                        seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                        score = scores[b_ix]
                        score_list.append(score)
                        native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                        if b_ix == 0 and j==0 and temp==temperatures[0]:
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(native_seq[start:end])
                                start = end
                            native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                            l0 = 0
                            for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                l0 += mc_length
                                native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                l0 += 1
                            f.write('>Native, score={}, visible_chains={}, masked_chains={}\n{}\n'.format(round(float(native_score[b_ix]),4), visible_list_list[0], masked_list_list[0], native_seq)) #write the native sequence
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(seq[start:end])
                            start = end

                        seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                        l0 = 0
                        for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                            l0 += mc_length
                            seq = seq[:l0] + '/' + seq[l0:]
                            l0 += 1
                        f.write('>T={}, sample={}, score={}, seq_recovery={}\n{}\n'.format(temp,b_ix,round(float(score),4),round(float(seq_recovery_rate.detach().cpu().numpy()), 4),seq)) #write generated sequence
        np.save(score_file, np.array(score_list, np.float32))
        t1 = time.time()
        dt = round(float(t1-t0), 4)
        num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
        total_length = X.shape[1]
        print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')
        
