import numpy as np
from scvi import REGISTRY_KEYS
from scvi.dataloaders import AnnDataLoader
import torch
import yaml
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import wandb
from sctree.utils.training_utils import predict
from sctree.utils.utils import cluster_acc, dendrogram_purity, leaf_purity
from sctree.utils.model_utils import construct_data_tree_wandb, construct_data_tree_samples

def evaluate(dataloader, model, config):   
	print("\n" * 2)
	print("Evaluation")
	print("\n" * 2)

	# Predict clusters
	dataloader = AnnDataLoader(model.adata_manager, batch_size = config['training']['batch_size'], shuffle=False)
	model_base = model.module
	prob_leaves = predict(dataloader, model_base, 'prob_leaves')
	yy = np.squeeze(np.argmax(prob_leaves, axis=-1)).numpy()
	y_true = dataloader.dataset.dataset.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY).flatten()
	
	# Determine indeces of samples that fall into each leaf for DP&LP
	leaves = model_base.compute_leaves()
	ind_samples_of_leaves = []
	for i in range(len(leaves)):
		ind_samples_of_leaves.append([leaves[i]['node'],np.where(yy==i)[0]])
	
	# Calculate standard metrics and  confusion matrix
	acc, idx = cluster_acc(y_true, yy, return_index=True)
	swap = dict(zip(range(len(idx)), idx))
	nmi = normalized_mutual_info_score(y_true, yy)
	ari = adjusted_rand_score(y_true, yy)
	dp = dendrogram_purity(model_base.tree, y_true, ind_samples_of_leaves)
	lp = leaf_purity(model_base.tree, y_true, ind_samples_of_leaves)

	# Create objects to store clustering
	y_wandb = np.array([swap[i] for i in yy], dtype=np.uint8)
	data_tree_wandb = construct_data_tree_wandb(model_base, y_predicted=yy, y_true=y_true, n_leaves=prob_leaves.shape[1])
	data_tree_samples = construct_data_tree_samples(model_base, y_predicted=yy, n_leaves=prob_leaves.shape[1])

	# Log to wandb
	wandb.log({"Test/Accuracy": acc, "Test/Normalized Mutual Information": nmi, "Test/Adjusted Rand Index": ari, "Test/Dendrogram Purity": dp, "Test/Leaf Purity": lp})
	wandb.log({"Test/Confusion Matrix": wandb.plot.confusion_matrix(probs=None,
																	y_true=y_true, preds=y_wandb,
																	class_names=range(len(idx)))})
	table = wandb.Table(columns=["node_id", "node_name", "parent", "size"], data=data_tree_wandb)
	fields = {"node_name": "node_name", "node_id": "node_id", "parent": "parent", "size": "size"}
	dendro = wandb.plot_table(vega_spec_name="stacey/flat_tree", data_table=table, fields=fields)
	wandb.log({"dendogram_final": dendro})

	# Store model and results
	if config['globals']['save_model']:
		experiment_path = config['globals']['experiment_path']
		print("\nSaving weights at ", experiment_path)
		torch.save(model_base.state_dict(), experiment_path/'model_weights.pt')
		with open(experiment_path / 'c_test.npy', 'wb') as save_file:
			np.save(save_file, prob_leaves)
		with open(experiment_path / 'data_tree_samples.npy', 'wb') as save_file:
			np.save(save_file, data_tree_samples)
		with open(experiment_path / 'data_tree_wandb.npy', 'wb') as save_file:
			np.save(save_file, data_tree_wandb)
		with open(experiment_path / 'config.yaml', 'w', encoding='utf8') as outfile:
			yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

	print(np.unique(yy, return_counts=True))
	print("Accuracy:", acc)
	print("Normalized Mutual Information:", nmi)
	print("Adjusted Rand Index:", ari)
	print("Dendrogram Purity:", dp)
	print("Leaf Purity:", lp)
	return