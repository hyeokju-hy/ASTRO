import numpy as np
from scipy.sparse import coo_matrix
import math
import sys
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.MONET import MONET
from models.Diffusion import Diffusion
from utility.load_data import Data, DiffusionData
from utility.utility_functions import create_adj_mat, normalize_adj_mat, set_seed

class MOON(nn.Module): # Modality-aware COnditiOnal DiffusioN-based Graph Refinement Module
    def __init__(self, data_generator:Data, args:dict):
        super(MOON, self).__init__()
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items
        self.steps = args['steps']
        self.max_del_ratio = args['max_del_ratio']
        self.diffusion_decay = args['diffusion_decay']
        self.inter_matrix = data_generator.R.tocoo()
        # add edges to original adjacency matrix -> input for diffusion
        self.inter_matrix_v, self.inter_matrix_t = None, None
        self.diffusion_lr = eval(args['diffusion_lr']) if type(args['diffusion_lr']) == str else args['diffusion_lr']
        
        # diffusion_v: diffusion model for generating visual modality based graph
        set_seed(args['seed'])
        self.diffusion_v = Diffusion(args, self.inter_matrix.A.T)
        self.diffusion_v.cuda()
        self.diffusion_v_optimizer = Adam(self.diffusion_v.parameters(),lr=self.diffusion_lr) 
        
        # diffusion_t: diffusion model for generating text modality based graph
        set_seed(args['seed'])
        self.diffusion_t = Diffusion(args, self.inter_matrix.A.T)
        self.diffusion_t.cuda()
        self.diffusion_t_optimizer = Adam(self.diffusion_t.parameters(), lr=self.diffusion_lr)
        
        diffusion_train_dataset = DiffusionData(torch.FloatTensor(self.inter_matrix.A.T))
        self.diffusion_train_loader = DataLoader(diffusion_train_dataset, batch_size=args['diffusion_batch_size'], shuffle=True)
        self.score_v = torch.tensor(self.inter_matrix.A.T, dtype=torch.float)
        self.score_t = torch.tensor(self.inter_matrix.A.T, dtype=torch.float)
        self.del_threshold = 0.8
        self.del_threshold_v, self.del_threshold_t = 0.8, 0.8
        
    def train_diffusion(self, diffusion:Diffusion, optimizer:Adam, user_emb:torch.Tensor, item_emb:torch.Tensor):
        diffusion.train()
        total_diffusion_loss = 0
        for (iids, item_vecs) in self.diffusion_train_loader:
            iids = iids.cuda()
            item_vecs = item_vecs.cuda()
            optimizer.zero_grad()
            batch_diffusion_loss = diffusion.calculate_loss(item_vecs, user_emb, item_emb, iids)
            batch_diffusion_loss.backward()
            optimizer.step()
            total_diffusion_loss += batch_diffusion_loss.cpu().item()
        
        total_diffusion_loss /= len(self.diffusion_train_loader)
        
        if math.isnan(total_diffusion_loss):
            print("ERROR: Diffusion loss is nan.")
            sys.exit()
        return total_diffusion_loss
    
    def generate_graph(self, diffusion:Diffusion, score:torch.Tensor, user_emb:torch.Tensor, item_emb:torch.Tensor, del_threshold:float):
        diffusion.eval()
        all_prediction = []
        data = torch.tensor(self.inter_matrix.A.T, dtype=torch.float)
        generate_batch_size = 32
        num_batches = (self.n_items + generate_batch_size - 1) // generate_batch_size
        
        for batch_idx in range(num_batches):
            start = batch_idx * generate_batch_size
            end = min(start + generate_batch_size, self.n_items)
            
            data_batch = data[start:end]
            score_batch = score[start:end]
            
            binary_tensor = data_batch.cuda()
            score_tensor = score_batch.cuda()
            idx_tensor = torch.arange(start, end, device=binary_tensor.device)
            
            with torch.no_grad():
                prediction = diffusion.p_sample(binary_tensor, idx_tensor, user_emb, item_emb, self.steps, self.steps).sigmoid()
            avg_prediction = self.diffusion_decay * score_tensor + (1 - self.diffusion_decay) * prediction
            all_prediction.append(avg_prediction.cpu())

        new_score = torch.cat(all_prediction, dim=0)

        # (user, item) transpose
        data_matrix, score_matrix = data.T, new_score.T

        existing_users, existing_items = torch.where(data_matrix != 0)
        existing_scores = score_matrix[existing_users, existing_items]
        del_idx_candidates = torch.where(existing_scores < del_threshold)[0]

        max_deletions = int(len(existing_users) * self.max_del_ratio)
        num_del_candidates = len(del_idx_candidates)

        if num_del_candidates > max_deletions:
            del_idx = del_idx_candidates[torch.argsort(existing_scores[del_idx_candidates].cuda()).cpu()[:max_deletions]]
        else:
            del_idx = del_idx_candidates
        
        del_idx = del_idx.cpu()
        mask = torch.isin(torch.arange(len(existing_users)), del_idx)
        new_users, new_items = existing_users[~mask], existing_items[~mask]
        
        # add new edges whose score are top-k, k = # of deleted edges
        num_edges_to_add = len(del_idx)
        candidate_users, candidate_items = torch.where(data_matrix == 0)
        candidate_scores = score_matrix[candidate_users, candidate_items].cuda()
        topk_idx = torch.argsort(candidate_scores, descending=True)[:num_edges_to_add]
        add_users = candidate_users.cuda()[topk_idx]
        add_items = candidate_items.cuda()[topk_idx]
        
        final_users = torch.cat([new_users, add_users.cpu()], dim=0).numpy()
        final_items = torch.cat([new_items, add_items.cpu()], dim=0).numpy()
        
        # del_threshold tune
        if num_del_candidates == 0:
            del_threshold = torch.min(existing_scores).cpu().item() + 0.01
        elif num_del_candidates < max_deletions:
            pass
        else: 
            del_threshold = max(del_threshold*0.99,0.45)
        
        return final_users, final_items, new_score, del_threshold
    
    def forward(self, user_emb_v_frozen, item_emb_v_frozen, user_emb_t_frozen, item_emb_t_frozen):
        diffusion_v_loss = self.train_diffusion(self.diffusion_v, self.diffusion_v_optimizer, user_emb_v_frozen, item_emb_v_frozen)
        diffusion_t_loss = self.train_diffusion(self.diffusion_t, self.diffusion_t_optimizer, user_emb_t_frozen, item_emb_t_frozen)
        diffusion_loss = diffusion_v_loss + diffusion_t_loss
        
        row_v, col_v, self.score_v, self.del_threshold_v = self.generate_graph(self.diffusion_v, self.score_v, user_emb_v_frozen, item_emb_v_frozen, self.del_threshold_v)
        new_inter_matrix_v = coo_matrix((np.ones_like(row_v), (row_v, col_v)), shape=(self.n_users, self.n_items))
        refined_adj_matrix_v = create_adj_mat(new_inter_matrix_v)
        moon_matrix_v = normalize_adj_mat(refined_adj_matrix_v).coalesce().cuda()

        row_t, col_t, self.score_t, self.del_threshold_t = self.generate_graph(self.diffusion_t, self.score_t, user_emb_t_frozen, item_emb_t_frozen, self.del_threshold_t)
        new_inter_matrix_t = coo_matrix((np.ones_like(row_t), (row_t, col_t)), shape=(self.n_users, self.n_items))
        refined_adj_matrix_t = create_adj_mat(new_inter_matrix_t)
        moon_matrix_t = normalize_adj_mat(refined_adj_matrix_t).coalesce().cuda()
        
        return diffusion_loss, moon_matrix_v, moon_matrix_t
    
class MARS: # ModAlity-awaRe Similarity-based Graph Refinement Module
    def __init__(self, data_generator:Data, args:dict, adj_matrix:coo_matrix, rating_matrix:torch.FloatTensor, image_feats:np.ndarray, text_feats:np.ndarray):
        self.data_generator = data_generator
        self.args = args
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items
        self.n_layers = args['n_layers']
        self.has_norm = args['has_norm']
        self.feat_embed_dim = args['feat_embed_dim']
        self.adj_matrix = adj_matrix
        self.rating_matrix = rating_matrix
        self.image_feats = image_feats
        self.text_feats = text_feats
        self.alpha = args['alpha']
        self.ta_weight = args['ta_weight']
        self.inter_matrix = data_generator.R.tocoo()
        self.ratio = args['ratio']
        
    def refine_graphs(self):
        image_preference, text_preference, image_embedding, text_embedding = self.load_pretrained_embeddings()
        inter_matrix_v = self.refine_edges(image_preference, image_embedding)
        inter_matrix_t = self.refine_edges(text_preference, text_embedding)
        refined_adj_matrix_v = create_adj_mat(inter_matrix_v)
        refined_adj_matrix_t = create_adj_mat(inter_matrix_t)
        mars_matrix_v = normalize_adj_mat(refined_adj_matrix_v).coalesce().cuda()
        mars_matrix_t = normalize_adj_mat(refined_adj_matrix_t).coalesce().cuda()
        return mars_matrix_v, mars_matrix_t

    def load_pretrained_embeddings(self):
        # using propagated embedding
        pretrained_model_path = 'ASTRO/codes/pretrained_model_' + self.args['dataset']
        pretrained_model = MONET(self.n_users, self.n_items, self.n_layers, self.has_norm, self.feat_embed_dim, self.adj_matrix, self.rating_matrix, self.image_feats, self.text_feats, self.alpha, self.ta_weight)
        model_name = list(torch.load(pretrained_model_path, map_location='cpu', weights_only=True).keys())[0]
        statedict = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)[model_name]
                
        pretrained_model.load_state_dict(statedict)
        pretrained_model.cuda()
        user_emb, item_emb = pretrained_model()
        image_preference, text_preference = torch.split(user_emb, [self.feat_embed_dim, self.feat_embed_dim], dim=1)
        image_embedding, text_embedding = torch.split(item_emb, [self.feat_embed_dim, self.feat_embed_dim], dim=1)
        
        return image_preference, text_preference, image_embedding, text_embedding
        
    def refine_edges(self, user_emb:torch.Tensor, item_emb:torch.Tensor):
        item_degree = torch.tensor(np.array(self.inter_matrix.sum(axis=0)).flatten(), dtype=torch.long)
        n_adds = (item_degree.float() * self.ratio).long()
        if self.ratio == 0 or n_adds.sum().item() == 0:
            print('Do not add edges')
            return self.inter_matrix
        
        user_emb_norm = F.normalize(user_emb, p=2, dim=1).cpu()
        item_emb_norm = F.normalize(item_emb, p=2, dim=1).cpu()
        cos_sim = torch.clamp(user_emb_norm @ item_emb_norm.T, min=-1, max=1) # shape: (n_users, n_items)
        users = torch.tensor(self.inter_matrix.row, dtype=torch.long)
        items = torch.tensor(self.inter_matrix.col, dtype=torch.long)
        cos_sim[users, items] = -1 # ignore existing edgs
        
        new_users = torch.tensor([], dtype=torch.long)
        new_items = torch.tensor([], dtype=torch.long)
        
        for i in tqdm(range(self.n_items), desc='refining edges'):
            candidates = cos_sim[:, i]
            topk_val, topk_idx = torch.topk(candidates, k=n_adds[i], largest=True)
            
            add_users = topk_idx[torch.where(topk_val > 0)[0]].long()
            new_users = torch.cat([new_users, add_users], dim=0)
            add_items = torch.full(add_users.shape, i)
            new_items = torch.cat([new_items, add_items], dim=0)

        print(f"number of added edges:{len(new_users)}")
        new_users = torch.cat([users, new_users], dim=0)
        new_items = torch.cat([items, new_items], dim=0)
        new_users = new_users.numpy()
        new_items = new_items.numpy()
        new_data = np.ones_like(new_users)
        new_inter_matrix = coo_matrix((new_data, (new_users, new_items)), shape=(self.n_users, self.n_items))
        
        return new_inter_matrix

class ASTRO(nn.Module):
    def __init__(self, data_generator:Data, args:dict, image_feats:np.ndarray, text_feats:np.ndarray):
        super(ASTRO, self).__init__()
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items
        self.n_layers = args['n_layers']
        self.alpha = args['alpha']
        self.has_norm = args['has_norm']
        self.feat_embed_dim = args['feat_embed_dim']
        self.ta_weight = args['ta_weight']
        self.ratio = args['ratio']
        self.cl_temp = args['cl_temp']
        self.adj_matrix = data_generator.adj_matrix
        self.norm_adj_matrix = normalize_adj_mat(self.adj_matrix).coalesce().cuda()
        self.nonzero_idx = data_generator.nonzero_idx
        nonzero_idx = torch.tensor(self.nonzero_idx).cuda().long().T
        self.rating_matrix = torch.sparse_coo_tensor(nonzero_idx, torch.ones((nonzero_idx.size(1))).cuda(), (self.n_users, self.n_items)).to_dense().cuda()
        self.image_feats = torch.tensor(image_feats, dtype=torch.float).cuda()
        self.text_feats = torch.tensor(text_feats, dtype=torch.float).cuda()

        self.image_preference = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.feat_embed_dim)
        self.text_preference = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.feat_embed_dim)
        nn.init.xavier_uniform_(self.image_preference.weight)
        nn.init.xavier_uniform_(self.text_preference.weight)
        self.image_embedding = nn.Embedding.from_pretrained(torch.tensor(image_feats, dtype=torch.float), freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(torch.tensor(text_feats, dtype=torch.float), freeze=True)
        self.image_trs = nn.Linear(image_feats.shape[1], self.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], self.feat_embed_dim)
        
        self.moon_matrix_v, self.moon_matrix_t = self.norm_adj_matrix, self.norm_adj_matrix

        self.mars = MARS(data_generator, args, self.adj_matrix, self.rating_matrix, image_feats, text_feats)
        self.mars_matrix_v, self.mars_matrix_t = self.mars.refine_graphs()
    
    def forward(self, view=False):
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
            if not view:
                side_image_emb = torch.sparse.mm(self.moon_matrix_v, ego_image_emb)
                side_text_emb = torch.sparse.mm(self.moon_matrix_t, ego_text_emb)
            else: 
                side_image_emb = torch.sparse.mm(self.mars_matrix_v, ego_image_emb)
                side_text_emb = torch.sparse.mm(self.mars_matrix_t, ego_text_emb)
                
            ego_image_emb = side_image_emb + self.alpha * ego_image_emb
            ego_text_emb = side_text_emb + self.alpha * ego_text_emb

        final_image_preference, final_image_emb = torch.split(ego_image_emb, [self.n_users, self.n_items], dim=0)
        final_text_preference, final_text_emb = torch.split(ego_text_emb, [self.n_users, self.n_items], dim=0)

        items = torch.cat([final_image_emb, final_text_emb], dim=1)
        user_preference = torch.cat([final_image_preference, final_text_preference], dim=1)

        return user_preference, items

    def calculate_loss(self, users, pos_items, neg_items, target_aware):
        # main embedding: emb1 -> BPR Loss
        # view: emb2 -> CL loss with emb1
        user_emb1, item_emb1 = self.forward(view=False) # moon
        user_emb2, item_emb2 = self.forward(view=True) # mars
        current_user_emb = user_emb1[users]
        pos_item_emb = item_emb1[pos_items]
        neg_item_emb = item_emb1[neg_items]

        if target_aware:
            # target-aware
            item_item = torch.mm(item_emb1, item_emb1.T)
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
                pos_target_user_alpha, item_emb1
            )  # (batch_size, dim)
            neg_target_user = torch.mm(
                neg_target_user_alpha, item_emb1
            )  # (batch_size, dim)

            # predictor
            pos_scores = (1 - self.ta_weight) * torch.sum(
                torch.mul(current_user_emb, pos_item_emb), dim=1
            ) + self.ta_weight * torch.sum(torch.mul(pos_target_user, pos_item_emb), dim=1)
            neg_scores = (1 - self.ta_weight) * torch.sum(
                torch.mul(current_user_emb, neg_item_emb), dim=1
            ) + self.ta_weight * torch.sum(torch.mul(neg_target_user, neg_item_emb), dim=1)
        else:
            pos_scores = torch.sum(torch.mul(current_user_emb, pos_item_emb), dim=1)
            neg_scores = torch.sum(torch.mul(current_user_emb, neg_item_emb), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        regularizer = (
            torch.pow(user_emb1, 2).sum() + 
            torch.pow(item_emb1, 2).sum()
        )
        reg_loss = regularizer / len(users)
        
        pos_items = torch.tensor(pos_items, dtype=torch.long).cuda()
        neg_items = torch.tensor(neg_items, dtype=torch.long).cuda()
        items = torch.cat([pos_items, neg_items], dim=0)
        
        cl_loss = self.infonce_loss(user_emb1, item_emb1, user_emb2, item_emb2, users, items)
        
        del user_emb1, item_emb1, user_emb2, item_emb2
        torch.cuda.empty_cache()

        return mf_loss, reg_loss, cl_loss
    
    def infonce_loss(self, user_emb1, item_emb1, user_emb2, item_emb2, users, items):
        norm_user_emb1 = F.normalize(user_emb1, p=2, dim=1)
        norm_item_emb1 = F.normalize(item_emb1, p=2, dim=1)
        norm_user_emb2 = F.normalize(user_emb2, p=2, dim=1)
        norm_item_emb2 = F.normalize(item_emb2, p=2, dim=1)
        
        pos_score_user = torch.exp(torch.sum(torch.mul(norm_user_emb1[users], norm_user_emb2[users]), dim=1) / self.cl_temp)
        pos_score_item = torch.exp(torch.sum(torch.mul(norm_item_emb1[items], norm_item_emb2[items]), dim=1) / self.cl_temp)

        neg_score_user = torch.sum(torch.exp(norm_user_emb1[users] @ norm_user_emb2[users].T / self.cl_temp), dim=1)
        neg_score_item = torch.sum(torch.exp(norm_item_emb1[items] @ norm_item_emb2[items].T / self.cl_temp), dim=1)

        cl_loss = -(1/2) * (torch.mean(torch.log(pos_score_user / neg_score_user)) + torch.mean(torch.log(pos_score_item / neg_score_item)))
        return cl_loss