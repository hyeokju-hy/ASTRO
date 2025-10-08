import torch
import numpy as np
import random
import torch.nn.functional as F
from scipy.sparse import coo_matrix, lil_matrix, diags
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def set_device(gpu_id:int):
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    return device

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed:", seed)
    
def get_eval_log(Ks:list, ret, is_val=True):
    eval_log = {}
    if type(Ks) == str:
        Ks = eval(Ks)
    for idx, k in enumerate(Ks):
        if is_val:
            eval_log[f"Val recall@{k}"] = float(f"{ret['recall'][idx]:.5f}")
            eval_log[f"Val precision@{k}"] = float(f"{ret['precision'][idx]:.5f}")
            eval_log[f"Val hit@{k}"] = float(f"{ret['hit_ratio'][idx]:.5f}")
            eval_log[f"Val ndcg@{k}"] = float(f"{ret['ndcg'][idx]:.5f}")
        else:
            eval_log[f"Final recall@{k}"] = float(f"{ret['recall'][idx]:.5f}")
            eval_log[f"Final precision@{k}"] = float(f"{ret['precision'][idx]:.5f}")
            eval_log[f"Final hit@{k}"] = float(f"{ret['hit_ratio'][idx]:.5f}")
            eval_log[f"Final ndcg@{k}"] = float(f"{ret['ndcg'][idx]:.5f}")
    return eval_log

def normalize_adj_mat(adj_matrix:coo_matrix):
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
    
def create_adj_mat(inter_matrix:coo_matrix):
    num_users, num_items = inter_matrix.shape
    adj_matrix = lil_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    R = inter_matrix.tolil()
    adj_matrix[:num_users, num_users:] = R
    adj_matrix[num_users:, :num_users] = R.T
    adj_matrix = adj_matrix.tocoo()
    return adj_matrix 