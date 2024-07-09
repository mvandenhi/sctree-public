import scanpy as sc
from models.sctree import scTree
from utils.utils import setup
from utils.eval_utils import evaluate
import wandb
from utils.utils import reset_random_seeds
import random
import time


def main():
    config = setup()

######### SWEEP STUFF #########
    config['training']['lr'] = wandb.config.lr
    config['training']['max_depth'] = wandb.config.max_depth
    config['training']['latent_dim'] = wandb.config.latent_dim
    config['training']['mlp_layers'] = wandb.config.mlp_layers
    config['training']['kl_start'] = wandb.config.kl_start
    config['training']['decoder_n_layers'] = wandb.config.decoder_n_layers
    config['training']['splitting_criterion'] = wandb.config.splitting_criterion
    config['training']['encoder'] = wandb.config.encoder
    config['globals']['seed'] = wandb.config.seed
    # random.seed(int(time.time()))
    # reset_random_seeds(random.randint(0, 1000000))
    reset_random_seeds(wandb.config.seed)

######### SWEEP STUFF #########

    adata = sc.read_h5ad(f"data/{config['data']['data_name']}.h5ad")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    try: # TODO: @Florian, is there a smarter way to do this for all datasets?
        scTree.setup_anndata(adata, layer="counts", labels_key="celltype", batch_key="sample")
    except:
        scTree.setup_anndata(adata, layer="counts", labels_key="celltype")
    model = scTree(adata, config['training'])
    model.train(config['training'])

    # Evaluate clustering
    evaluate(adata, model, config)

######### SWEEP STUFF #########
# Define sweep config
sweep_configuration = {
    "method": "grid",
    "name": "grid-splitting-criterion",
    "metric": {"goal": "maximize", "name": "Test/Dendrogram Purity"},
    "parameters": {
        "lr": {"value": 0.0001},
        "seed": {"values": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]},
        "latent_dim": {"value": 16},
        "mlp_layers": {"value": 128},
        "kl_start": {"value": 1},
        "max_depth": {"values": [4,5,6,7]},
        "splitting_criterion": {"values": ['n_samples', 'reconstruction', 'weighted_reconstruction', 'weights']},
        "decoder_n_layers": {"value": 1},
        "encoder": {"value": 'mlp'},
    },
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="scTreeVAE-sweep")
wandb.agent(sweep_id, function=main)
######### SWEEP STUFF #########


# if __name__ == '__main__':
# 	main()