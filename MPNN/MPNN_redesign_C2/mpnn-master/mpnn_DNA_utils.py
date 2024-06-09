from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

import copy
import torch.nn as nn
import torch.nn.functional as F
import random

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
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av



class StructureDataset():
    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYXatcgdryu-'):
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

                # Convert raw coords to np arrays
                #for key, val in entry['coords'].items():
                #    entry['coords'][key] = np.asarray(val)

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        if True:
                            self.data.append(entry)
                        else:
                            discard_count['bad_seq_length'] += 1
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
            'full': (3, num_positional_embeddings + 2*num_chain_embeddings),
            'dist': (6, num_positional_embeddings + num_rbf),
            'hbonds': (3, 2 * num_positional_embeddings),
        }

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.dropout = nn.Dropout(dropout)
        self.linear_chains = nn.Linear(2, num_chain_embeddings)
        self.embeddings_type = nn.Linear(5, num_chain_embeddings)
        # Normalization and embedding
        node_in, edge_in = self.feature_dimensions[features_type]
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.edge_pp = nn.Linear(num_rbf*15 + 7*11, edge_features, bias=True)
        self.edge_nn = nn.Linear(num_rbf*7 + 7*5, edge_features, bias=True)
        self.edge_pn = nn.Linear(num_rbf*20 + 7*6, edge_features, bias=True)
        self.edge_np = nn.Linear(num_rbf*20 + 7*6, edge_features, bias=True)
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

    def _get_orientations_full(self, A, B, C, D, E, F_, E_idx, eps=1e-6):
        v1 = A-B #[B, L, 3]
        v2 = C-B #[B, L, 3]
        e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps) #[B, L, 3]
        u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[:,:,None]*e1) #[B, L, 3]
        e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
        e3 = torch.cross(e1, e2, dim=-1)
        O = torch.cat([e1[:,:,:,None], e2[:,:,:,None], e3[:,:,:,None]], axis=-1) #[L,3,3] - rotation matrix
        O = O.view(list(O.shape[:2]) + [9])

        v1 = D-E #[B, L, 3]
        v2 = F_-E #[B, L, 3]
        e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps) #[B, L, 3]
        u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[:,:,None]*e1) #[B, L, 3]
        e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
        e3 = torch.cross(e1, e2, dim=-1)
        O_ = torch.cat([e1[:,:,:,None], e2[:,:,:,None], e3[:,:,:,None]], axis=-1) #[L,3,3] - rotation matrix
        O_ = O_.view(list(O_.shape[:2]) + [9])

        O_neighbors = gather_nodes(O_, E_idx)
        E_neighbors = gather_nodes(E, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3,3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

        # Rotate into local reference frames
        dX = E_neighbors - B.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
        Q = self._quaternions(R)
        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1)
        return O_features

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


    def _orientations_coarse(self, X, E_idx, eps=1e-6):
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

    def forward(self, X, L, mask, residue_idx, dihedral_mask, chain_labels, type_encoding_all, top_k_sample=False):
        """ Featurize coordinates as an attributed graph """
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        protein_mask = (type_encoding_all == 0).float()
        not_protein_mask = 1.0 - protein_mask
        protein_protein_mask = protein_mask[:,:,None]*protein_mask[:,None,:]
        not_protein_not_protein_mask = not_protein_mask[:,:,None]*not_protein_mask[:,None,:]
        protein_not_protein_mask = protein_mask[:,:,None]*not_protein_mask[:,None,:] #[B, L, L]
        not_protein_protein_mask = not_protein_mask[:,:,None]*protein_mask[:,None,:] #[B, L, L]

        DNA_mask = (type_encoding_all == 1).float()
        RNA_mask = (type_encoding_all == 2).float()
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        # Build k-Nearest Neighbors graph
        X_coarse = X[:,:,1,:]*protein_mask[:,:,None] + X[:,:,4,:]*not_protein_mask[:,:,None] #[B,L,3] - CA for proteins, P for DNA/RNA
        D_neighbors, E_idx, mask_neighbors = self._dist(X_coarse, mask, top_k_sample)

        pp_mask = gather_edges(protein_protein_mask[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]
        nn_mask = gather_edges(not_protein_not_protein_mask[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]
        pn_mask = gather_edges(protein_not_protein_mask[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]
        np_mask = gather_edges(not_protein_protein_mask[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        # Pairwise features
        AD_features, O_features = self._orientations_coarse(X_coarse, E_idx)

        #atoms for protein backbone
        N = X[:,:,0,:]
        Ca = X[:,:,1,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]

        #atoms for DNA/RNA
        P = X[:,:,4]
        OP1 = X[:,:,5]
        OP2 = X[:,:,6]
        O5p = X[:,:,7]
        C5p = X[:,:,8]
        C4p = X[:,:,9]
        O4p = X[:,:,10]
        C3p = X[:,:,11]
        O3p = X[:,:,12]
        C2p = X[:,:,13]
        C1p = X[:,:,14]
        N1 =  X[:,:,15]

        RBF_all_nn = []
        RBF_all_nn.append(self._rbf(D_neighbors)) #P-P
        RBF_all_nn.append(self._get_rbf(N1, N1, E_idx))
        RBF_all_nn.append(self._get_rbf(C4p, C4p, E_idx))
        RBF_all_nn.append(self._get_rbf(OP1, P, E_idx))
        RBF_all_nn.append(self._get_rbf(OP2, P, E_idx))
        RBF_all_nn.append(self._get_rbf(N1, P, E_idx))
        RBF_all_nn.append(self._get_rbf(C4p, P, E_idx))
        RBF_all_nn = torch.cat(tuple(RBF_all_nn), dim=-1)

        O_features_list_nn = []
        O_features_list_nn.append(O_features) #P-P-P
        O_features_list_nn.append(self._get_orientations(OP1, P, OP2, E_idx))
        O_features_list_nn.append(self._get_orientations(O4p, C1p, C2p, E_idx))
        O_features_list_nn.append(self._get_orientations(O4p, C1p, N1, E_idx))
        O_features_list_nn.append(self._get_orientations(OP1, P, O5p, E_idx))
        O_features_all_nn = torch.cat(tuple(O_features_list_nn), dim=-1)

        RBF_all_pn = []
        RBF_all_pn.append(self._get_rbf(N, P, E_idx))
        RBF_all_pn.append(self._get_rbf(Ca, P, E_idx))
        RBF_all_pn.append(self._get_rbf(C, P, E_idx))
        RBF_all_pn.append(self._get_rbf(O, P, E_idx))
        RBF_all_pn.append(self._get_rbf(Cb, P, E_idx))
        RBF_all_pn.append(self._get_rbf(N, N1, E_idx))
        RBF_all_pn.append(self._get_rbf(Ca, N1, E_idx))
        RBF_all_pn.append(self._get_rbf(C, N1, E_idx))
        RBF_all_pn.append(self._get_rbf(O, N1, E_idx))
        RBF_all_pn.append(self._get_rbf(Cb, N1, E_idx))
        RBF_all_pn.append(self._get_rbf(N, OP1, E_idx))
        RBF_all_pn.append(self._get_rbf(Ca, OP1, E_idx))
        RBF_all_pn.append(self._get_rbf(C, OP1, E_idx))
        RBF_all_pn.append(self._get_rbf(O, OP1, E_idx))
        RBF_all_pn.append(self._get_rbf(Cb, OP1, E_idx))
        RBF_all_pn.append(self._get_rbf(N, OP2, E_idx))
        RBF_all_pn.append(self._get_rbf(Ca, OP2, E_idx))
        RBF_all_pn.append(self._get_rbf(C, OP2, E_idx))
        RBF_all_pn.append(self._get_rbf(O, OP2, E_idx))
        RBF_all_pn.append(self._get_rbf(Cb, OP2, E_idx))
        RBF_all_pn = torch.cat(tuple(RBF_all_pn), dim=-1)

        O_features_list_pn = []
        O_features_list_pn.append(self._get_orientations_full(N, Ca, C, OP1, P, OP2, E_idx))
        O_features_list_pn.append(self._get_orientations_full(N, Ca, C, O4p, C1p, C2p, E_idx))
        O_features_list_pn.append(self._get_orientations_full(N, Ca, C, O4p, N1, C1p, E_idx))
        O_features_list_pn.append(self._get_orientations_full(C, Ca, Cb, OP1, P, OP2, E_idx))
        O_features_list_pn.append(self._get_orientations_full(C, Ca, Cb, O4p, C1p, C2p, E_idx))
        O_features_list_pn.append(self._get_orientations_full(C, Ca, Cb, O4p, N1, C1p, E_idx))
        O_features_all_pn = torch.cat(tuple(O_features_list_pn), dim=-1)

        RBF_all_np = []
        RBF_all_np.append(self._get_rbf(P, N, E_idx))
        RBF_all_np.append(self._get_rbf(P, Ca, E_idx))
        RBF_all_np.append(self._get_rbf(P, C, E_idx))
        RBF_all_np.append(self._get_rbf(P, O, E_idx))
        RBF_all_np.append(self._get_rbf(P, Cb, E_idx))
        RBF_all_np.append(self._get_rbf(N1, N, E_idx))
        RBF_all_np.append(self._get_rbf(N1, Ca, E_idx))
        RBF_all_np.append(self._get_rbf(N1, C, E_idx))
        RBF_all_np.append(self._get_rbf(N1, O, E_idx))
        RBF_all_np.append(self._get_rbf(N1, Cb, E_idx))
        RBF_all_np.append(self._get_rbf(OP1, N, E_idx))
        RBF_all_np.append(self._get_rbf(OP1, Ca, E_idx))
        RBF_all_np.append(self._get_rbf(OP1, C, E_idx))
        RBF_all_np.append(self._get_rbf(OP1, O, E_idx))
        RBF_all_np.append(self._get_rbf(OP1, Cb, E_idx))
        RBF_all_np.append(self._get_rbf(OP2, N, E_idx))
        RBF_all_np.append(self._get_rbf(OP2, Ca, E_idx))
        RBF_all_np.append(self._get_rbf(OP2, C, E_idx))
        RBF_all_np.append(self._get_rbf(OP2, O, E_idx))
        RBF_all_np.append(self._get_rbf(OP2, Cb, E_idx))
        RBF_all_np = torch.cat(tuple(RBF_all_np), dim=-1)

        O_features_list_np = []
        O_features_list_np.append(self._get_orientations_full(OP1, P, OP2, N, Ca, C, E_idx))
        O_features_list_np.append(self._get_orientations_full(O4p, C1p, C2p, N, Ca, C, E_idx))
        O_features_list_np.append(self._get_orientations_full(O4p, N1, C1p, N, Ca, C, E_idx))
        O_features_list_np.append(self._get_orientations_full(OP1, P, OP2, C, Ca, Cb, E_idx))
        O_features_list_np.append(self._get_orientations_full(O4p, C1p, OP2, C, Ca, Cb, E_idx))
        O_features_list_np.append(self._get_orientations_full(O4p, N1, C1p, C, Ca, Cb, E_idx))
        O_features_all_np = torch.cat(tuple(O_features_list_np), dim=-1)

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

        offset_type = type_encoding_all[:,:,None] - type_encoding_all[:,None,:] + 2
        offset_type = torch.nn.functional.one_hot(gather_edges(offset_type[:,:,:,None], E_idx)[:,:,:,0], 5).float() #[B, L, K]   
        E_type_positional = self.embeddings_type(offset_type)

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_chains = torch.nn.functional.one_hot(E_chains, 2).float()
        E_chains = self.linear_chains(E_chains)

        if self.features_type == 'coarse':
            # Coarse backbone features
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'full':
            # Full backbone angles
            V = AD_features
            E = torch.cat((E_chains, E_positional, E_type_positional), -1)

        E_pp = self.edge_pp(torch.cat((RBF_all, O_features_all), -1))*pp_mask[:,:,:,None]
        E_nn = self.edge_nn(torch.cat((RBF_all_nn, O_features_all_nn), -1))*nn_mask[:,:,:,None]
        E_pn = self.edge_pn(torch.cat((RBF_all_pn, O_features_all_pn), -1))*pn_mask[:,:,:,None]
        E_np = self.edge_np(torch.cat((RBF_all_np, O_features_all_np), -1))*np_mask[:,:,:,None]
        E = self.edge_embedding(E)
        E = E + E_pp + E_nn + E_pn + E_np
        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.norm_edges(E)

        return V, E, E_idx


def _scores(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1,log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores

def _S_to_seq(S, mask):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
    return seq
