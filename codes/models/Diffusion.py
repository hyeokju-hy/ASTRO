import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

# 1. MLP
class MLP(nn.Module):
    def __init__(self, args:dict, in_dims:list, out_dims:list):
        super(MLP, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_embed_dim = args['time_embed_dim']
        self.emb_layer = nn.Linear(self.time_embed_dim, self.time_embed_dim)

        in_dims_temp = self.in_dims
        out_dims_temp = self.out_dims
        self.in_layers = nn.ModuleList(nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]))
        self.out_layers = nn.ModuleList(nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]))

        self.drop = nn.Dropout(0.5)
        self.init_weights()
        self.act_func = nn.ReLU()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            layer:nn.Linear
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, item_emb=None): 
        # x: (batch_size, embed_size)
        half = self.time_embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half).cuda()
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_embed_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)

        time_emb = self.emb_layer(time_emb)
        x = F.normalize(x)
        x = self.drop(x)
        
        if item_emb == None:
            h = torch.cat([x, time_emb], dim=-1).float()
        else:
            h = torch.cat([x, time_emb, item_emb], dim=-1).float()
        
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = self.act_func(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = self.act_func(h)
        return h
    
# 2. Diffusion
class Diffusion(nn.Module):
    def __init__(self, args:dict, inter_matrix:np.ndarray):
        super(Diffusion, self).__init__()
        self.inter_matrix = inter_matrix
        self.user_degree = self.inter_matrix.sum(axis=0)
        self.item_degree = self.inter_matrix.sum(axis=1)
        self.n_users = self.inter_matrix.shape[1]
        self.noise_scale = args['noise_scale']
        self.noise_min = args['noise_min']
        self.noise_max = args['noise_max']
        self.steps = args['steps']
        self.ddim = args['ddim']
        self.hidden_units = args['hidden_units']
        
        in_dims = [self.n_users + args['feat_embed_dim'] + args['time_embed_dim'], self.hidden_units]
        out_dims = [self.hidden_units, self.n_users]
        
        self.MLP = MLP(args, in_dims, out_dims)
        self.base_betas = self.get_base_betas()
    
    def get_base_betas(self):
        # base scheduler of diffusion model: beta_t = noise_scale * (beta_min + (t-1)/(T-1)(beta_max - beta_min))
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        base_betas = np.linspace(start, end, self.steps, dtype=np.float32)
        base_betas = torch.tensor(base_betas, dtype=torch.float32).cuda()
        base_betas[0] = 0.00001
        return base_betas
    
    def get_batch_betas(self, user_emb:torch.Tensor, item_emb:torch.Tensor, iids:torch.LongTensor):
        base_betas = self.base_betas.unsqueeze(1).unsqueeze(1).expand(self.steps, len(iids), self.n_users) # (steps, batch_size, n_users)
        
        with torch.no_grad():
            user_emb_norm = F.normalize(user_emb, p=2, dim=1)
            item_emb_norm = F.normalize(item_emb[iids], p=2, dim=1)
            preference_score = torch.clamp(item_emb_norm @ user_emb_norm.T, min=-1, max=1) # (batch_size, n_users)
        inter = torch.tensor(self.inter_matrix[iids.cpu().numpy()]).cuda()
        preference_score = torch.where(inter == 1, preference_score, -preference_score)
        gamma = 1 - 0.01 * torch.exp(3 * preference_score)
        
        refined_betas = gamma.unsqueeze(0) * base_betas # (steps, batch_size, n_users)
        return refined_betas
                
    def calculate_batch_for_diffusion(self, user_emb:torch.Tensor, item_emb:torch.Tensor, iids:torch.LongTensor):
        betas = self.get_batch_betas(user_emb, item_emb, iids)
        alphas = 1.0 - betas  # (steps, batch_size, n_users)
        sqrt_recip_alphas=torch.sqrt(1/alphas)
        alphas_cumprod = torch.cumprod(alphas, axis=0) # (steps, batch_size, n_users)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        ones_tensor = torch.ones((1, alphas_cumprod.size(1), alphas_cumprod.size(2)), dtype=torch.float).cuda()
        zeros_tensor = torch.zeros((1, alphas_cumprod.size(1), alphas_cumprod.size(2)), dtype=torch.float).cuda()
        alphas_cumprod_prev = torch.cat([ones_tensor, alphas_cumprod[:-1,:,:]], dim=0)
        alphas_cumprod_next = torch.cat([alphas_cumprod[1:,:,:], zeros_tensor], dim=0)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        posterior_mean_coef2 = ((1.0 - alphas_cumprod_prev)* torch.sqrt(alphas)/ (1.0 - alphas_cumprod))   
        fast_posterior_coef2=(torch.sqrt(1.0 - alphas_cumprod_prev)/sqrt_one_minus_alphas_cumprod)
        fast_posterior_coef3=(sqrt_alphas_cumprod* torch.sqrt(1.0 - alphas_cumprod_prev)/sqrt_one_minus_alphas_cumprod)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        
        variables = {
            "sqrt_recip_alphas": sqrt_recip_alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "alphas_cumprod_next": alphas_cumprod_next,
            "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
            "log_one_minus_alphas_cumprod": log_one_minus_alphas_cumprod,
            "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
            "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
            "posterior_mean_coef1": posterior_mean_coef1,
            "posterior_mean_coef2": posterior_mean_coef2,
            "fast_posterior_coef2": fast_posterior_coef2,
            "fast_posterior_coef3": fast_posterior_coef3,
            "posterior_variance": posterior_variance
        }
        return variables
    
    def calculate_loss(self, x_start:torch.Tensor, user_emb:torch.Tensor, item_emb:torch.Tensor, iids:torch.LongTensor):
        variables = self.calculate_batch_for_diffusion(user_emb, item_emb, iids)
        batch_size = x_start.size(0)
        ts, pt = self.sample_timesteps(batch_size)
        noise = torch.randn_like(x_start)
        # forward process
        x_t = self.q_sample(variables, x_start, ts, noise)
        # reverse process
        pred = self.MLP(x_t, ts, item_emb[iids])
        loss = (x_start - pred) ** 2
        weight = self.SNR(variables, ts - 1) - self.SNR(variables, ts)
        ts_expand = ts[:, None].expand(len(ts), self.n_users)
        weight = torch.where((ts_expand == 0), torch.tensor(1.0, dtype=weight.dtype).cuda(), weight)
        diffusion_loss = (torch.mean(weight * loss, dim=1) / pt).mean()
        #diffusion_loss = (torch.mean(loss, dim=1) / pt).mean()
        return diffusion_loss
    
    def SNR(self, variables, ts):
        """
        Compute the Signal-to-Noise Ratio for a single timestep.
        """
        alphas_cumprod = variables["alphas_cumprod"]
        return self.extract_from_tensor(alphas_cumprod, ts) / (1 - self.extract_from_tensor(alphas_cumprod,ts))
    
    def sample_timesteps(self, batch_size):
        ts = torch.randint(0, self.steps, (batch_size, ), device='cuda').long() # timesteps
        pt = torch.ones_like(ts, dtype=torch.float)
        return ts, pt
    
    def q_sample(self, variables, x_start, ts, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod = variables["sqrt_alphas_cumprod"]
        sqrt_one_minus_alphas_cumprod = variables["sqrt_one_minus_alphas_cumprod"]
        q_t = self.extract_from_tensor(sqrt_alphas_cumprod, ts) * x_start + self.extract_from_tensor(sqrt_one_minus_alphas_cumprod, ts) * noise
        return q_t
    
    def p_sample(self, x_start, iids:torch.LongTensor ,user_emb:torch.Tensor, item_emb:torch.Tensor, steps, noise_step, sampling_noise=True):
        assert steps <= self.steps, "Too much steps in inference."
        variables = self.calculate_batch_for_diffusion(user_emb, item_emb, iids)
        if noise_step == 0:
            x_t = x_start            
        else:
            t = torch.tensor([noise_step - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(variables, x_start, t)
        if self.ddim:
            reverse_step=int(steps/10)
        else:
            reverse_step=steps

        indices = list(range(reverse_step))[::-1]
        input_embed = item_emb[iids]
        
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            out = self.p_mean_variance(variables,x_t,t,input_embed)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))) )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * torch.log(out["variance"])) * noise
            else:
                x_t = out["mean"]
        return x_t
    
    def p_mean_variance(self,variables,x,t,batch_embed):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
    
        model_variance = variables["posterior_variance"]
        model_variance = self.extract_from_tensor(model_variance,t)

        model_output = self.MLP(x.to(batch_embed.dtype),t, batch_embed)
        
        pred_xstart = model_output
        if not self.ddim:
            model_mean, _ = self.q_posterior_mean_variance(variables=variables, x_start=pred_xstart, x_t=x, t=t)
        else:
            model_mean, _= self.fast_q_posterior_mean_variance(variables=variables, x_start=pred_xstart, x_t=x, t=t)
        
        return {"mean": model_mean, "variance": model_variance}#, "log_variance": model_log_variance}
    
    def q_posterior_mean_variance(self, variables, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean_coef1 = variables["posterior_mean_coef1"]
        posterior_mean_coef2 = variables["posterior_mean_coef2"]
        posterior_variance = variables["posterior_variance"]
        posterior_mean =self.extract_from_tensor(posterior_mean_coef1,t) * x_start+self.extract_from_tensor(posterior_mean_coef2,t)* x_t
        posterior_variance = self.extract_from_tensor(posterior_variance,t)

        return posterior_mean, posterior_variance#, posterior_log_variance_clipped  

    def fast_q_posterior_mean_variance(self, variables, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior with ddim
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        alphas_cumprod_prev = variables["alphas_cumprod_prev"]
        fast_posterior_coef2 = variables["fast_posterior_coef2"]
        fast_posterior_coef3 = variables["fast_posterior_coef3"]
        posterior_variance = variables["posterior_variance"]
        
        posterior_mean = self.extract_from_tensor(torch.sqrt(alphas_cumprod_prev),t)* x_start\
                        +self.extract_from_tensor(fast_posterior_coef2,t)*x_t-self.extract_from_tensor(fast_posterior_coef3,t)*x_start

        posterior_variance = self.extract_from_tensor(posterior_variance,t)
        
        return posterior_mean, posterior_variance#, posterior_log_variance_clipped
    
    def extract_from_tensor(self,A:torch.Tensor,C:torch.Tensor):
        return A[C, torch.arange(A.shape[1]).cuda(), :]