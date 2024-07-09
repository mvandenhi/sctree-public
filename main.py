import scanpy as sc
from sctree.models.sctree import scTree
import wandb
def main():

    wandb.init(
        project="sctree",
        mode="offline",
        entity="FlorianBarkmann"
    )

    adata = sc.read_h5ad(f"data/pbmc_subset.h5ad")
    sc.pp.highly_variable_genes(adata, n_top_genes=4000, subset=True, batch_key="Experiment")
    #scTree.setup_anndata(adata, layer="counts", labels_key="CellType", batch_key="Experiment", categorical_covariate_keys=["Method"])
    scTree.setup_anndata(adata, layer="counts", labels_key="CellType", batch_key="Experiment")

    model = scTree(adata, adata.obs["CellType"].nunique(), max_depth=6, decoder_n_layers=0)
    model.train()
    print(model.get_cluster_probabilities())


if __name__ == '__main__':
    main()