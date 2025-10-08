import numpy as np
from models.MONET import MONET
import torch
from torch import optim
import math
import sys
from utility.batch_test import test_torch
from utility.load_data import Data
from utility.utility_functions import get_eval_log, set_device, set_seed
from time import time
import wandb

class MONET_trainer: 
    def __init__(self, data_generator:Data, args:dict):
        self.args = args
        self.device = set_device(args['gpu_id'])
        self.data_generator = data_generator
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items
        self.n_batch = data_generator.n_train // args['batch_size'] + 1
        self.feat_embed_dim = args['feat_embed_dim']
        self.lr = eval(args['lr'])
        self.emb_dim = args['embed_size'] 
        self.batch_size = args['batch_size']
        self.n_layers = args['n_layers']
        self.has_norm = args['has_norm']
        self.decay = eval(args['regs'])
        self.alpha = args['alpha']
        self.beta = args['beta']
        self.dataset = args['dataset']
        self.target_aware = args['target_aware']
        self.nonzero_idx = data_generator.nonzero_idx
        nonzero_idx = torch.tensor(self.nonzero_idx).cuda().long().T
        self.rating_matrix = torch.sparse_coo_tensor(nonzero_idx, torch.ones((nonzero_idx.size(1))).cuda(), (self.n_users, self.n_items)).to_dense().cuda()
        self.inter_matrix = data_generator.R
        self.users_to_test = list(data_generator.test_set.keys())
        self.users_to_val = list(data_generator.val_set.keys())

        self.image_feats = np.load(args['data_path'] + '{}/image_feat.npy'.format(self.dataset))
        self.text_feats = np.load(args['data_path'] + '{}/text_feat.npy'.format(self.dataset))
        
        self.adj_matrix = data_generator.adj_matrix
        
        set_seed(args['seed'])
        self.model = MONET(self.n_users, self.n_items, self.n_layers, self.has_norm, self.feat_embed_dim, self.adj_matrix, self.rating_matrix, self.image_feats, self.text_feats, self.alpha, self.beta)
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model_name = f"{args['model_name']}_beta[{self.beta}]"
    
    def run_model(self):
        use_wandb, Ks = self.args['use_wandb'], eval(self.args['Ks'])
        if use_wandb:
            wandb.run.name = f"{self.model_name}"
            
        best_recall, stopping_step = 0, 0
        ret = self.test()
        
        print(ret)
        if use_wandb:
            eval_log = get_eval_log(Ks, ret, is_val=True)
            wandb.log(eval_log, step=0)
            
        if ret['recall'][0] > best_recall:
            best_recall = ret['recall'][0]
            self.save_model()
        for epoch in range(1, self.args['epoch'] + 1):
            t1 = time()
            train_log = self.train()
            print(f"Epoch {epoch}: time = {time() - t1:.1f}s, train loss = [{train_log['Total loss']:.5f} = {train_log['BPR loss']:.5f} + {train_log['Reg loss']:.5f}]")
            if use_wandb:
                wandb.log(train_log, step=epoch)
            
            if epoch % self.args['verbose'] == 0:
                t2 = time()
                ret = self.test()
                perf_str = (
                    f"Validation: time = [{time() - t2:.1f}], Ks = {Ks}, "
                    f"recall = {[float(f'{x:.5f}') for x in ret['recall']]}, "
                    f"precision = {[float(f'{x:.5f}') for x in ret['precision']]}, "
                    f"hit = {[float(f'{x:.5f}') for x in ret['hit_ratio']]}, "
                    f"ndcg = {[float(f'{x:.5f}') for x in ret['ndcg']]}"
                )
                print(perf_str)
                
                if ret['recall'][0] > best_recall:
                    best_recall = ret['recall'][0]
                    stopping_step = 0
                    print('Found better model.')
                    self.save_model()
                elif stopping_step < self.args['early_stopping_patience']:
                    stopping_step += 1
                    print(f'Early stopping steps: {stopping_step}')
                else:
                    print('Early Stop!')
                    break
                
                if use_wandb:
                    eval_log = get_eval_log(Ks, ret, is_val=True)
                    wandb.log(eval_log, step=epoch)
        
        # test model
        ret = self.test(is_val=False)
        print('Final result:', ret)
        if use_wandb:
            eval_log = get_eval_log(Ks, ret, is_val=False)
            wandb.log(eval_log)
            
    def train(self):
        total_loss, bpr_loss, reg_loss = 0, 0, 0
        
        for _ in range(self.n_batch):
            self.model.train() # set model into training mode
            self.optimizer.zero_grad()  
            user_emb, item_emb = self.model()
            users, pos_items, neg_items = self.data_generator.sample()
            
            batch_bpr_loss, batch_reg_loss = self.model.calculate_loss(user_emb, item_emb, users, pos_items, neg_items, self.target_aware)
            batch_reg_loss = batch_reg_loss * self.decay
            batch_loss = batch_bpr_loss + batch_reg_loss
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()

            total_loss = total_loss + batch_loss.cpu().item()
            bpr_loss = bpr_loss + batch_bpr_loss.cpu().item()
            reg_loss = reg_loss + batch_reg_loss.cpu().item()

            del user_emb, item_emb
            if self.device == torch.device('cuda'):
                torch.cuda.empty_cache()
            
        if math.isnan(total_loss):
            print("ERROR: loss is nan.")
            sys.exit()
        
        total_loss = total_loss / self.n_batch
        bpr_loss = bpr_loss / self.n_batch
        reg_loss = reg_loss / self.n_batch
        train_log = {"Total loss": total_loss, "BPR loss": bpr_loss, "Reg loss": reg_loss}
        return train_log

    def test(self, is_val=True):
        self.model.eval() # set model into evaluation mode
        with torch.no_grad():
            if is_val:
                ua_embeddings, ia_embeddings = self.model()
            else: 
                self.model = MONET(self.n_users, self.n_items, self.n_layers, self.has_norm, self.feat_embed_dim, self.adj_matrix, self.rating_matrix, self.image_feats, self.text_feats, self.alpha, self.beta)
                self.model.load_state_dict(torch.load('ASTRO/codes/saved_models/' + self.model_name, map_location='cpu', weights_only=True)[self.model_name])
                self.model.cuda()
                ua_embeddings, ia_embeddings = self.model()
        users = self.users_to_val if is_val else self.users_to_test
        result = test_torch(ua_embeddings, ia_embeddings, users, is_val, self.rating_matrix, self.beta, self.target_aware)
        return result
   
    def save_model(self):
        torch.save({self.model_name: self.model.state_dict()}, 'ASTRO/codes/saved_models/' + self.model_name)