from utility.parser import config, sweep_config
from utility.load_data import data_generator
import wandb
import importlib

# wandb project name
project_name = ''

def main():
    use_wandb = config['use_wandb']
    use_sweep = config['use_sweep']
    model_name = config['model_name']
    model = config['model']
    try:
        trainer_lib = importlib.import_module(name=f"trainers.{model}_trainer")
    except Exception as e:
        raise NotImplementedError(f"Trainer class is not defined, error message: {e}")
    
    if use_sweep and sweep_config == None:
        raise ValueError("Sweep config is not defined!")
    if use_wandb:
        wandb.init(project=project_name, config=config, name=model_name)
        if use_sweep: 
            config.update(dict(wandb.config))
            
    trainer_class = getattr(trainer_lib, f'{model}_trainer')
    trainer = trainer_class(data_generator=data_generator, args=config)

    trainer.run_model()
            
if __name__ == '__main__':  
    if config['use_sweep']:
        sweep_config['name'] = f"{config['model']}"
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id=sweep_id, function=main, project=project_name)
    else:
        main()