import numpy as np
import math
import sys
from time import time
import wandb
import torch
from torch import optim
from torch.optim import Adam
from models.ASTRO import ASTRO, MOON
from utility.utility_functions import set_device, set_seed, get_eval_log
from utility.batch_test import test_torch
from utility.load_data import Data

class ASTRO_trainer: 
    def __init__(self, data_generator:Data, args:dict):
        print('*' * 25, 'arguments', '*'  * 25)
        for key in args.keys():
            print(f"{key}: {args[key]}")
        print('*' * 25, 'arguments', '*'  * 25)
        # data settings
        self.device = set_device(args['gpu_id'])
        self.dataset = args['dataset']
        self.data_generator = data_generator
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items
        self.n_batch = data_generator.n_train // args['batch_size'] + 1
        self.nonzero_idx = data_generator.nonzero_idx
        nonzero_idx = torch.tensor(self.nonzero_idx).cuda().long().T
        self.rating_matrix = torch.sparse_coo_tensor(nonzero_idx, torch.ones((nonzero_idx.size(1))).cuda(), (self.n_users, self.n_items)).to_dense().cuda()
        self.inter_matrix = data_generator.R.tocoo()
        self.users_to_test = list(data_generator.test_set.keys())
        self.users_to_val = list(data_generator.val_set.keys())
        self.image_feats = np.load(args['data_path'] + '{}/image_feat.npy'.format(self.dataset))
        self.text_feats = np.load(args['data_path'] + '{}/text_feat.npy'.format(self.dataset))
        self.adj_matrix = data_generator.adj_matrix
        
        # argument settings
        # basic arguments
        self.args = args
        # MONET arguments
        self.feat_embed_dim = args['feat_embed_dim']
        self.lr = eval(args['lr']) if type(args['lr']) == str else args['lr']
        self.batch_size = args['batch_size']
        self.n_layers = args['n_layers']
        self.alpha = args['alpha']
        self.has_norm = args['has_norm']
        self.decay = eval(args['regs']) if type(args['regs']) == str else args['regs']
        self.ta_weight = args['ta_weight']
        self.target_aware = args['target_aware']
        self.cl_decay = args['cl_decay']
        self.ratio = args['ratio']
        # Diffusion arguments
        self.diffusion_decay = args['diffusion_decay']
        self.steps = args['steps']
        self.max_del_ratio = args['max_del_ratio']
        
        # model, optimizer settings    
        set_seed(args['seed'])
        self.model = ASTRO(self.data_generator, self.args, self.image_feats, self.text_feats)
        self.model.cuda()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler() 
        
        self.model_name = f"{args['model_name']}"
        print(f"saved model name: {self.model_name}")
        
        self.moon = MOON(self.data_generator, self.args)
        self.saved_moon_matrix_v, self.saved_moon_matrix_t = None, None
        
    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler
        
    def run_model(self):
        use_wandb, Ks = self.args['use_wandb'], eval(self.args['Ks'])
        if use_wandb:
            wandb.run.name = self.model_name
            
        best_recall, stopping_step = 0, 0
        
        ####################### evaluate #######################
        ret = self.test()
        print(ret)
        if use_wandb:
            eval_log = get_eval_log(Ks, ret, is_val=True)
            wandb.log(eval_log, step=0)
            
        if ret['recall'][0] > best_recall:
            best_recall = ret['recall'][0]
            self.save_model()
        ########################################################

        #score_v, score_t, del_threshold_v, del_threshold_t = self.moon.score_v, self.moon.score_t, self.moon.del_threshold, self.moon.del_threshold
        for epoch in range(1, self.args['epoch'] + 1):
            ############################################ train ############################################
            t1 = time()
            total_loss, bpr_loss, reg_loss, cl_loss = self.train_model()
            self.lr_scheduler.step()
            model_train_time = time() - t1
            train_info = f"Epoch {epoch}: time = {model_train_time:.1f}s, train loss = [{total_loss:.5f} = {bpr_loss:.5f} + {reg_loss:.5f} + {cl_loss:.5f}]"
            train_log = {"Total Loss": total_loss, "BPR Loss": bpr_loss, "Reg Loss": reg_loss, "CL Loss": cl_loss}
            
            with torch.no_grad():
                user_emb, item_emb = self.model(view=True)
                user_emb_v, user_emb_t = torch.split(user_emb, [self.feat_embed_dim, self.feat_embed_dim], dim=1)
                item_emb_v, item_emb_t = torch.split(item_emb, [self.feat_embed_dim, self.feat_embed_dim], dim=1)
                user_emb_v_frozen, item_emb_v_frozen = user_emb_v.clone(), item_emb_v.clone()
                user_emb_t_frozen, item_emb_t_frozen = user_emb_t.clone(), item_emb_t.clone()

            t2 = time()
            diffusion_loss, self.model.moon_matrix_v, self.model.moon_matrix_t = self.moon(user_emb_v_frozen, item_emb_v_frozen, user_emb_t_frozen, item_emb_t_frozen)
            diffusion_train_time = time() - t2
            
            total_time = model_train_time + diffusion_train_time            
            train_info = f"Epoch {epoch}: time = [{total_time:.1f} = {model_train_time:.1f} + {diffusion_train_time:.1f}]s, "
            train_info += f"train loss = [{total_loss:.5f} = {bpr_loss:.5f} + {reg_loss:.5f} + {cl_loss:.5f}], diffusion loss = [{diffusion_loss:.5f}]"
            train_log['Diffusion Loss'] = diffusion_loss
            
            print(train_info)
            if use_wandb:
                wandb.log(train_log, step=epoch)
            ###############################################################################################
            
            ########################################### evaluate ###########################################
            if epoch % self.args['verbose'] == 0 and epoch >= 60:
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
                    self.saved_moon_matrix_v = self.model.moon_matrix_v
                    self.saved_moon_matrix_t = self.model.moon_matrix_t
                elif stopping_step < self.args['early_stopping_patience']:
                    stopping_step += 1
                    print(f'Early stopping steps: {stopping_step}')
                else:
                    print('Early Stop!')
                    break
                
                if use_wandb:
                    eval_log = get_eval_log(Ks, ret, is_val=True)
                    wandb.log(eval_log, step=epoch)
            ###############################################################################################
        
        # test model
        ret = self.test(is_val=False)
        print('Final result:', ret)
        if use_wandb:
            eval_log = get_eval_log(Ks, ret, is_val=False)
            wandb.log(eval_log)
            
    def train_model(self):
        total_loss, bpr_loss, reg_loss, cl_loss = 0, 0, 0, 0
        
        for _ in range(self.n_batch):
            self.model.train() # set model into training mode
            self.optimizer.zero_grad()  
            
            users, pos_items, neg_items = self.data_generator.sample()
            
            batch_bpr_loss, batch_reg_loss, batch_cl_loss = self.model.calculate_loss(users, pos_items, neg_items, self.target_aware)
            batch_reg_loss = batch_reg_loss * self.decay
            batch_cl_loss = batch_cl_loss * self.cl_decay
            batch_loss = batch_bpr_loss + batch_reg_loss + batch_cl_loss
            
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()

            total_loss = total_loss + batch_loss.cpu().item()
            bpr_loss = bpr_loss + batch_bpr_loss.cpu().item()
            reg_loss = reg_loss + batch_reg_loss.cpu().item()
            cl_loss = cl_loss + batch_cl_loss.cpu().item()


        if math.isnan(total_loss):
            print("ERROR: loss is nan.")
            sys.exit()
        
        total_loss = total_loss / self.n_batch
        bpr_loss = bpr_loss / self.n_batch
        reg_loss = reg_loss / self.n_batch
        cl_loss = cl_loss / self.n_batch
        
        return total_loss, bpr_loss, reg_loss, cl_loss
    
    def test(self, is_val=True):
        self.model.eval() # set model into evaluation mode
        with torch.no_grad():
            if is_val:
                ua_embeddings, ia_embeddings = self.model() # MOON embeddings
            else: 
                self.model = ASTRO(self.data_generator, self.args, self.image_feats, self.text_feats)
                self.model.load_state_dict(torch.load('ASTRO/codes/saved_models/' + self.model_name, map_location='cpu', weights_only=True)[self.model_name])
                self.model.moon_matrix_v = self.saved_moon_matrix_v
                self.model.moon_matrix_t = self.saved_moon_matrix_t
                self.model.cuda()
                ua_embeddings, ia_embeddings = self.model() # MOON embeddings
        users = self.users_to_val if is_val else self.users_to_test
        result = test_torch(ua_embeddings, ia_embeddings, users, is_val, self.rating_matrix, self.ta_weight, self.target_aware)
        return result

    def save_model(self):
        torch.save({self.model_name: self.model.state_dict()}, 'ASTRO/codes/saved_models/' + self.model_name)