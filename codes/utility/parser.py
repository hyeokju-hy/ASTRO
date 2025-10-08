import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', nargs='?', default='ASTRO/data/',
                        help='Input data path.')
    parser.add_argument('--saved_model_path', default='ASTRO/codes/saved_models/',
                        help='Input path to save model.')
    parser.add_argument('--dataset', nargs='?', default='Beauty',
                        help='Choose a dataset from {Baby, Beauty, Toys_and_Games, MenClothing, WomenClothing}')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--core', type=int, default=5,
                        help='5-core for warm-start; 0-core for cold start.')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='') 
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--Ks', nargs='?', default='[10, 20]',
                        help='K value of precision/recall/ndcg @ k')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--model', type=str, default='ASTRO', help='select model')
    parser.add_argument('--seed', type=int, default=2048)
    parser.add_argument('--use_wandb', type=bool, default=False, help='use wandb')
    parser.add_argument('--use_sweep', type=bool, default=False, help='use sweep')
    return parser.parse_args()

train_args = vars(parse_args())

if train_args['model_name'] == None:
    raise ValueError('model_name is None!')
if not train_args['use_wandb'] and train_args['use_sweep']:
    raise TypeError('wandb should be used to use sweep!')

with open("./utility/config.yaml", "r") as f:
    config = yaml.safe_load(f)
config = config[train_args['model']]
train_args.update(config)
config = train_args
sweep_config = config.pop('sweep', None)