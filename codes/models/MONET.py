import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, diags
import numpy as np    
        
class MONET(nn.Module):
    def __init__(self,
                n_users:int,
                n_items:int,
                n_layers:int,
                has_norm:bool,
                feat_embed_dim:int,
                adj_matrix:coo_matrix,
                rating_matrix:torch.FloatTensor,
                image_feats:np.ndarray,
                text_feats:np.ndarray,
                alpha:float,
                beta:float
        ):
        super(MONET, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.has_norm = has_norm
        self.feat_embed_dim = feat_embed_dim
        self.adj_matrix = adj_matrix
        self.norm_adj_matrix = self.normalize_adj_mat(self.adj_matrix).coalesce().cuda()
        self.rating_matrix = rating_matrix
        self.image_feats = torch.tensor(image_feats, dtype=torch.float).cuda()
        self.text_feats = torch.tensor(text_feats, dtype=torch.float).cuda()
        self.alpha = alpha
        self.beta = beta

        self.image_preference = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.feat_embed_dim)
        self.text_preference = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.feat_embed_dim)
        nn.init.xavier_uniform_(self.image_preference.weight)
        nn.init.xavier_uniform_(self.text_preference.weight)
        self.image_embedding = nn.Embedding.from_pretrained(torch.tensor(image_feats, dtype=torch.float), freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(torch.tensor(text_feats, dtype=torch.float), freeze=True)
        self.image_trs = nn.Linear(image_feats.shape[1], self.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], self.feat_embed_dim)

    def normalize_adj_mat(self, adj_matrix:coo_matrix):
        rowsum = np.array(adj_matrix.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_matrix = diags(d_inv)
        
        sparse_mx = d_matrix.dot(adj_matrix)
        sparse_mx = sparse_mx.dot(d_matrix)
        sparse_mx = sparse_mx.tocoo().astype(np.float32)

        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)
    
    def forward(self):
        # MeGCN
        image_emb = self.image_trs(self.image_embedding.weight)
        text_emb = self.text_trs(self.text_embedding.weight)
        if self.has_norm:
            image_emb = F.normalize(image_emb)
            text_emb = F.normalize(text_emb)
        image_preference = self.image_preference.weight
        text_preference = self.text_preference.weight
        # propagate
        ego_image_emb = torch.cat([image_preference, image_emb], dim=0)
        ego_text_emb = torch.cat([text_preference, text_emb], dim=0)

        for layer in range(self.n_layers):
            side_image_emb = torch.sparse.mm(self.norm_adj_matrix, ego_image_emb)
            side_text_emb = torch.sparse.mm(self.norm_adj_matrix, ego_text_emb)

            ego_image_emb = side_image_emb + self.alpha * ego_image_emb
            ego_text_emb = side_text_emb + self.alpha * ego_text_emb

        final_image_preference, final_image_emb = torch.split(ego_image_emb, [self.n_users, self.n_items], dim=0)
        final_text_preference, final_text_emb = torch.split(ego_text_emb, [self.n_users, self.n_items], dim=0)

        items = torch.cat([final_image_emb, final_text_emb], dim=1)
        user_preference = torch.cat([final_image_preference, final_text_preference], dim=1)

        return user_preference, items
    
    def calculate_loss(self, user_emb, item_emb, users, pos_items, neg_items, target_aware):
        current_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]

        if target_aware:
            # target-aware
            item_item = torch.mm(item_emb, item_emb.T)
            pos_item_query = item_item[pos_items, :]  # (batch_size, n_items)
            neg_item_query = item_item[neg_items, :]  # (batch_size, n_items)
            pos_target_user_alpha = torch.softmax(
                torch.multiply(pos_item_query, self.rating_matrix[users, :]).masked_fill(
                    self.rating_matrix[users, :] == 0, -1e9
                ),
                dim=1,
            )  # (batch_size, n_items)
            neg_target_user_alpha = torch.softmax(
                torch.multiply(neg_item_query, self.rating_matrix[users, :]).masked_fill(
                    self.rating_matrix[users, :] == 0, -1e9
                ),
                dim=1,
            )  # (batch_size, n_items)
            pos_target_user = torch.mm(
                pos_target_user_alpha, item_emb
            )  # (batch_size, dim)
            neg_target_user = torch.mm(
                neg_target_user_alpha, item_emb
            )  # (batch_size, dim)

            # predictor
            pos_scores = (1 - self.beta) * torch.sum(
                torch.mul(current_user_emb, pos_item_emb), dim=1
            ) + self.beta * torch.sum(torch.mul(pos_target_user, pos_item_emb), dim=1)
            neg_scores = (1 - self.beta) * torch.sum(
                torch.mul(current_user_emb, neg_item_emb), dim=1
            ) + self.beta * torch.sum(torch.mul(neg_target_user, neg_item_emb), dim=1)
        else:
            pos_scores = torch.sum(torch.mul(current_user_emb, pos_item_emb), dim=1)
            neg_scores = torch.sum(torch.mul(current_user_emb, neg_item_emb), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        regularizer = (
            1.0 / 2 * (pos_item_emb**2).sum()
            + 1.0 / 2 * (neg_item_emb**2).sum()
            + 1.0 / 2 * (current_user_emb**2).sum()
        )
        reg_loss = regularizer / pos_item_emb.size(0)


        return mf_loss, reg_loss
