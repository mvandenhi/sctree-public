from typing import Optional, List, Literal

import numpy as np
from copy import deepcopy
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalObsField,
    CategoricalJointObsField,
    NumericalJointObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.model.base import BaseModelClass
from scvi.utils import setup_anndata_dsp
from torch import optim

from sctree.module.treemodule import TreeModule, SmallTreeModule

from sctree.utils.training_utils import (
    train_one_epoch,
    validate_one_epoch,
    get_optimizer,
    get_ind_small_tree,
    predict,
    AnnealKLCallback,
    compute_growing_leaf,
    compute_pruning_leaf,
    Custom_Metrics,
    compute_leaves,
)

from sctree.utils.model_utils import return_list_tree, construct_data_tree_samples
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


_GLOBAL = "global"
import wandb


class scTree(BaseModelClass):
    _module_cls = TreeModule

    def __init__(
        self,
        adata: AnnData,
        n_cluster: int,
        max_depth: int,
        n_latent: int = 10,
        n_hidden: int = 128,
        dispersion: Literal["gene", "cluster"] = "gene",
        likelihood: Literal["NB", "ZINB"] = "NB",
        encoder_arch: str = "mlp",
        kl_start: float = 0.001,
        lambda_diva_kl: float = 1.0,
        lambda_diva_ce: float = 1.0,
        decoder_n_layers: int = 1,
        router_n_layers: int = 1,
        router_dropout: float = 0.1,
        mlp_n_layers: int = 1,
        batch_correction: Literal[_GLOBAL, "cluster", "diva"] = _GLOBAL,
    ):
        super().__init__(adata)

        self.n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        self.max_depth = max_depth
        self.depth = 1
        self.n_cluster = n_cluster
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.kl_start = kl_start
        self.decoder_n_layers = decoder_n_layers
        self.router_n_layers = router_n_layers
        self.router_dropout = router_dropout
        self.dispersion = dispersion
        self.likelihood = likelihood
        self.batch_correction = batch_correction
        self.mlp_n_layers = mlp_n_layers
        self.module = self._module_cls(
            inital_depth=self.depth,
            max_depth=self.max_depth,
            n_input=self.summary_stats.n_vars,
            n_latent=self.n_latent,
            n_batch=self.summary_stats.n_batch,
            n_cats_per_cov=self.n_cats_per_cov,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_hidden=self.n_hidden,
            dispersion=self.dispersion,
            likelihood=self.likelihood,
            kl_start=kl_start,
            lambda_diva_kl=lambda_diva_kl,
            lambda_diva_ce=lambda_diva_ce,
            encoder_arch=encoder_arch,
            decoder_n_layers=self.decoder_n_layers,
            router_n_layers=self.router_n_layers,
            router_dropout=self.router_dropout,
            mlp_n_layers=self.mlp_n_layers,
            batch_correction=batch_correction,
        )

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """ """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        # register new fields if the adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = 50,
        intermediate_epochs: int = 40,
        finetuning_epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 128,
        annealing_strategy: Literal["linear", "const"] = "linear",
        splitting_criterion: str = "n_samples",
        kl_start: float = 0.001,
    ):

        # Initialize training helpers
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize schedulers
        self.module.to(device)

        train_dataloader = AnnDataLoader(
            self.adata_manager, batch_size=batch_size, shuffle=True, drop_last=True
        )
        train_dataloader_unshuffled = AnnDataLoader(
            self.adata_manager, batch_size=batch_size, shuffle=False
        )

        metrics_calc_train = Custom_Metrics(device)
        metrics_calc_val = Custom_Metrics(device)
        # Training the initial split
        self.train_tree(
            train_dataloader,
            train_dataloader_unshuffled,
            lr,
            weight_decay,
            max_epochs,
            annealing_strategy,
            kl_start,
            device,
            metrics_calc_train,
            metrics_calc_val,
        )

        # Start the growing loop of the tree
        # Compute metrics and set node.expand False for the nodes that should not grow
        # This loop goes layer-wise
        initial_depth = self.depth
        max_depth = self.max_depth
        grow = initial_depth < max_depth
        growing_iterations = 0
        while grow and growing_iterations < 150:

            # full model finetuning during growing every 3 splits
            if growing_iterations != 0 and growing_iterations % 3 == 0:
                if splitting_criterion == "grow_elbo_reduction":
                    if allow_intermediate_finetuning:
                        print("\nTree intermediate finetuning\n")
                        for leaf in leaves:
                            leaf["node"].expand = True
                        self.train_tree(
                            train_dataloader,
                            train_dataloader_unshuffled,
                            lr,
                            weight_decay,
                            intermediate_epochs,
                            annealing_strategy,
                            kl_start,
                            device,
                            metrics_calc_train,
                            metrics_calc_val,
                        )
                    else:
                        pass

                else:
                    print("\nTree intermediate finetuning\n")
                    self.train_tree(
                        train_dataloader,
                        train_dataloader_unshuffled,
                        lr,
                        weight_decay,
                        intermediate_epochs,
                        annealing_strategy,
                        kl_start,
                        device,
                        metrics_calc_train,
                        metrics_calc_val,
                    )
            # extract information of leaves
            node_leaves_train, rec_loss_train, rec_loss_train_unweighted, elbo = (
                predict(
                    train_dataloader_unshuffled,
                    self.module,
                    "node_leaves",
                    "rec_loss_leafwise",
                    "return_recloss_leafwise_unweighted",
                    "elbo",
                )
            )

            # compute which leaf to split
            ind_leaf, leaf, n_effective_leaves = compute_growing_leaf(
                train_dataloader_unshuffled,
                self.module,
                node_leaves_train,
                rec_loss_train,
                rec_loss_train_unweighted,
                elbo,
                max_depth,
                batch_size,
                max_leaves=self.n_cluster,
                splitting_criterion=splitting_criterion,
            )
            if ind_leaf == None:
                break
            else:
                if splitting_criterion not in (
                    "grow_all",
                    "grow_all_rec_diff",
                    "grow_all_rec_diff_short",
                ):
                    print(
                        "\nGrowing tree: Leaf %d at depth %d\n"
                        % (ind_leaf, leaf["depth"])
                    )
                    depth, node = leaf["depth"], leaf["node"]

            if splitting_criterion in (
                "grow_all",
                "grow_all_rec_diff",
                "grow_all_rec_diff_short",
            ):
                n_samples_leaves = ind_leaf
                nonzero_ids = [i != None for i in n_samples_leaves]
                leaves_id = torch.arange(len(n_samples_leaves))[nonzero_ids]
                potential_small_models = []
                small_model_elbo = []
                small_model_rec_diff = []
                leaves = compute_leaves(self.module.tree)
                for i in range(n_effective_leaves):
                    # get subset of data that has high prob. of falling in subtree
                    leaf_id = leaves_id[i]
                    leaf = leaves[leaf_id]
                    depth, node = leaf["depth"], leaf["node"]
                    ind_train = get_ind_small_tree(
                        node_leaves_train[leaf_id], n_effective_leaves
                    )

                    if len(ind_train) < batch_size or depth == max_depth:
                        potential_small_models.append(None)
                        small_model_rec_diff.append(0)
                        small_model_elbo.append(torch.full((1,), torch.inf).item())
                    else:
                        # get subset of data that has high prob. of falling in subtree
                        if splitting_criterion == "grow_all_rec_diff_short":
                            train_epochs = 10
                        else:
                            train_epochs = max_epochs

                        train_small = AnnDataLoader(
                            self.adata_manager,
                            indices=ind_train,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                        )
                        self.train_subtree(
                            train_small,
                            train_epochs,
                            lr,
                            weight_decay,
                            annealing_strategy,
                            kl_start,
                            depth,
                            metrics_calc_train,
                            metrics_calc_val,
                            device,
                            leaf_id,
                        )
                        potential_small_models.append(self.small_model)

                        # Compute ELBO and reconstruction difference between both leaves
                        copy_tree = deepcopy(self.module)
                        copy_leaves = compute_leaves(copy_tree.tree)
                        copy_leaf = copy_leaves[leaf_id]
                        copy_node = copy_leaf["node"]
                        copy_tree.attach_smalltree(copy_node, self.small_model)
                        elbo = predict(train_dataloader_unshuffled, copy_tree, "elbo")
                        small_model_elbo.append(elbo.mean())

                        # Computing how different the two reconstructions are to see how similar the two learnt clusters are
                        leafwise_recloss_unweighted, prob_leaves = predict(
                            train_small,
                            copy_tree,
                            "return_recloss_leafwise_unweighted",
                            "prob_leaves",
                        )  # Note this is on subset only, as this is where we want to compare distributions
                        leaf_ids = torch.zeros(2, dtype=torch.long)
                        for i in range(2):
                            for j in range(len(copy_tree.decoders)):
                                if copy_tree.decoders[j] is None:
                                    continue
                                elif (
                                    self.small_model.decoders[i]
                                    is copy_tree.decoders[j]
                                ):
                                    break
                                else:
                                    leaf_ids[i] += 1

                        if (
                            prob_leaves[:, leaf_ids].mean(0) < 0.05
                        ).any():  # One leaf collapsed
                            small_model_rec_diff.append(0)
                        else:
                            probs = prob_leaves[:, leaf_ids].sum(1)
                            rec_losses = (
                                leafwise_recloss_unweighted[:, leaf_ids[0]]
                                - leafwise_recloss_unweighted[:, leaf_ids[1]]
                            ).abs()
                            rec_diff = (rec_losses * probs).sum() / probs.sum()
                            small_model_rec_diff.append(rec_diff.item())

                # attach smalltree to full tree by assigning decisions and adding new children nodes to full tree
                if splitting_criterion == "grow_all":
                    leaf_id_final = np.argmin(small_model_elbo)
                elif splitting_criterion in (
                    "grow_all_rec_diff",
                    "grow_all_rec_diff_short",
                ):
                    leaf_id_final = np.argmax(small_model_rec_diff)
                    # If no more growing to be done
                    if all(leaf == 0 for leaf in small_model_rec_diff):
                        break

                ind_leaf = leaves_id[leaf_id_final]
                leaf = leaves[ind_leaf]
                depth, node = leaf["depth"], leaf["node"]
                print("\nGrowing tree: Leaf %d at depth %d\n" % (ind_leaf, depth))

                if splitting_criterion == "grow_all_rec_diff_short":
                    train_epochs = max_epochs - 10
                    ind_train = get_ind_small_tree(
                        node_leaves_train[ind_leaf], n_effective_leaves
                    )
                    train_small = AnnDataLoader(
                        self.adata_manager,
                        indices=ind_train,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                    )
                    self.train_subtree(
                        train_small,
                        train_epochs,
                        lr,
                        weight_decay,
                        annealing_strategy,
                        kl_start,
                        depth,
                        metrics_calc_train,
                        metrics_calc_val,
                        device,
                        ind_leaf,
                        pretrained_small_model=potential_small_models[leaf_id_final],
                    )
                if self.batch_correction == _GLOBAL:
                    self.module.batch_decoder.thaw()
                self.module.attach_smalltree(
                    node, potential_small_models[leaf_id_final]
                )
                if n_effective_leaves + 1 == self.n_cluster:
                    max_growth = self.stopping_criterion(
                        train_dataloader_unshuffled,
                        max_depth,
                        batch_size,
                        splitting_criterion,
                    )
                    if max_growth is True:
                        break

            elif splitting_criterion == "grow_elbo_reduction":
                allow_intermediate_finetuning = False
                ind_train = get_ind_small_tree(
                    node_leaves_train[ind_leaf], n_effective_leaves
                )
                # get subset of data that has high prob. of falling in subtree
                train_small = AnnDataLoader(
                    self.adata_manager,
                    indices=ind_train,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                )

                # Allow for 5 initializations to prevent collapse
                i = 0
                while True and i < 5:
                    copy_tree = deepcopy(self)
                    copy_leaves = compute_leaves(copy_tree.module.tree)
                    copy_leaf = copy_leaves[ind_leaf]
                    copy_node = copy_leaf["node"]
                    copy_tree.train_subtree(
                        train_small,
                        max_epochs // 2,
                        lr,
                        weight_decay,
                        annealing_strategy,
                        kl_start,
                        depth,
                        metrics_calc_train,
                        metrics_calc_val,
                        device,
                        ind_leaf,
                    )
                    if copy_tree.batch_correction == _GLOBAL:
                        copy_tree.module.batch_decoder.thaw()
                    copy_tree.module.attach_smalltree(copy_node, copy_tree.small_model)
                    elbo_post, prob_leaves = predict(
                        train_dataloader_unshuffled,
                        copy_tree.module,
                        "elbo",
                        "prob_leaves",
                    )
                    print("Before training: ", prob_leaves.mean(0))
                    copy_tree.train_tree(
                        train_dataloader,
                        train_dataloader_unshuffled,
                        lr,
                        weight_decay,
                        max_epochs // 2,
                        annealing_strategy,
                        kl_start,
                        device,
                        metrics_calc_train,
                        metrics_calc_val,
                    )
                    elbo_pre = predict(
                        train_dataloader_unshuffled, self.module, "elbo"
                    ).mean()
                    elbo_post, prob_leaves = predict(
                        train_dataloader_unshuffled,
                        copy_tree.module,
                        "elbo",
                        "prob_leaves",
                    )
                    print("After training: ", prob_leaves.mean(0))
                    leaf_ids = torch.zeros(2, dtype=torch.long)
                    for i in range(2):
                        for j in range(len(copy_tree.module.decoders)):
                            if copy_tree.module.decoders[j] is None:
                                continue
                            elif (
                                copy_tree.small_model.decoders[i]
                                is copy_tree.module.decoders[j]
                            ):
                                break
                            else:
                                leaf_ids[i] += 1
                    if (
                        elbo_post.mean() > elbo_pre
                        or (
                            prob_leaves[:, leaf_ids].mean(0)
                            / prob_leaves[:, leaf_ids].mean(0).sum()
                            < 0.05
                        ).any()
                    ):
                        # If ELBO doesn't improve or collapsed, try again
                        i += 1
                        print("Iteration", i)
                    else:
                        self = deepcopy(copy_tree)
                        break

                else:
                    # Don't grow if ELBO wasn't improved or one leaf collapsed
                    print(
                        "ELBO of leaf did not improve. Before, ELBO was %.2f"
                        % elbo_pre,
                        "now it is %.2f" % elbo_post.mean(),
                    )
                    node.expand = False

                # Check if reached the max number of effective leaves before finetuning unnecessarily
                if n_effective_leaves + 1 == self.n_cluster:
                    max_growth = self.stopping_criterion(
                        train_dataloader_unshuffled,
                        max_depth,
                        batch_size,
                        splitting_criterion,
                    )
                    if max_growth is True:
                        break

            else:
                ind_train = get_ind_small_tree(
                    node_leaves_train[ind_leaf], n_effective_leaves
                )
                # get subset of data that has high prob. of falling in subtree
                train_small = AnnDataLoader(
                    self.adata_manager,
                    indices=ind_train,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                )

                self.train_subtree(
                    train_small,
                    max_epochs,
                    lr,
                    weight_decay,
                    annealing_strategy,
                    kl_start,
                    depth,
                    metrics_calc_train,
                    metrics_calc_val,
                    device,
                    ind_leaf,
                )

                # attach smalltree to full tree by assigning decisions and adding new children nodes to full tree
                if self.batch_correction == _GLOBAL:
                    self.module.batch_decoder.thaw()

                self.module.attach_smalltree(node, self.small_model)

                # Check if reached the max number of effective leaves before finetuning unnecessarily
                if n_effective_leaves + 1 == self.n_cluster:
                    max_growth = self.stopping_criterion(
                        train_dataloader_unshuffled,
                        max_depth,
                        batch_size,
                        splitting_criterion,
                    )
                    if max_growth is True:
                        break

            growing_iterations += 1

        # check whether we prune and log pre-pruning dendrogram

        node_leaves_test = predict(
            train_dataloader_unshuffled, self.module, "node_leaves"
        )

        if len(node_leaves_test) < 2:
            prune = False
        else:
            prune = True

        # prune the tree
        while prune:
            # check pruning conditions
            node_leaves_train = predict(
                train_dataloader_unshuffled, self.module, "node_leaves"
            )
            ind_leaf, leaf = compute_pruning_leaf(
                self.module, node_leaves_train, self.n_cluster
            )

            if ind_leaf == None:
                print("\nPruning finished!\n")
                break
            else:
                # prune leaves and internal nodes without children
                print(f"\nPruning leaf {ind_leaf}!\n")
                current_node = leaf["node"]
                while all(
                    child is None for child in [current_node.left, current_node.right]
                ):
                    if current_node.parent is not None:
                        parent = current_node.parent
                    # root does not get pruned
                    else:
                        break
                    parent.prune_child(current_node)
                    current_node = parent

                # reinitialize model
                transformations, routers, denses, decoders, routers_q = (
                    return_list_tree(self.module.tree)
                )
                self.module.decisions_q = routers_q
                self.module.transformations = transformations
                self.module.decisions = routers
                self.module.denses = denses
                self.module.decoders = decoders
                self.module.depth = self.module.compute_depth()

        print(
            "\n*****************model depth %d******************\n"
            % (self.module.depth)
        )
        print("\n*****************model finetuning******************\n")

        self.train_tree(
            train_dataloader,
            train_dataloader_unshuffled,
            lr,
            weight_decay,
            finetuning_epochs,
            annealing_strategy,
            kl_start,
            device,
            metrics_calc_train,
            metrics_calc_val,
        )

    def get_tree(self, return_labels: bool = False):

        dataloader = AnnDataLoader(self.adata_manager, shuffle=False)
        prob_leaves = predict(dataloader, self.module, "prob_leaves")
        yy = np.squeeze(np.argmax(prob_leaves, axis=-1)).numpy()
        data_tree_samples = construct_data_tree_samples(
            self.module, y_predicted=yy, n_leaves=prob_leaves.shape[1]
        )
        if not return_labels:
            return data_tree_samples
        else:
            return yy, data_tree_samples

    def get_cluster_probabilities(self):
        dataloader = AnnDataLoader(self.adata_manager, shuffle=False)
        prob_leaves = predict(dataloader, self.module, "prob_leaves")
        return prob_leaves

    def train_subtree(
        self,
        train_small,
        max_epochs,
        lr,
        weight_decay,
        annealing_strategy,
        kl_start,
        depth,
        metrics_calc_train,
        metrics_calc_val,
        device,
        ind_leaf,
        pretrained_small_model=None,
    ):

        # preparation for the smalltree training
        # initialize the smalltree
        if pretrained_small_model is not None:
            self.small_model = pretrained_small_model
        else:
            self.small_model = SmallTreeModule(
                depth=depth + 1,
                n_input=self.summary_stats.n_vars,
                n_latent=self.n_latent,
                px_r=self.module.px_r,
                batch_decoder=self.module.batch_decoder,
                n_batch=self.summary_stats.n_batch,
                encoded_size_gen=self.n_latent,
                n_hidden=self.n_hidden,
                n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
                n_cats_per_cov=self.n_cats_per_cov,
                kl_start=self.kl_start,
                decoder_n_layers=self.decoder_n_layers,
                router_n_layers=self.router_n_layers,
                router_dropout=self.router_dropout,
                mlp_n_layers=self.mlp_n_layers,
                likelihood=self.likelihood,
            )
            self.small_model.to(device)
            if self.dispersion == "gene":
                self.small_model.freeze_dispersion()
            if self.batch_correction == _GLOBAL:
                self.module.batch_decoder.freeze()

        # Optimizer for smalltree
        optimizer = get_optimizer(self.small_model, lr, weight_decay)
        # Initialize schedulers
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        alpha_scheduler = AnnealKLCallback(
            self.small_model,
            epochs=max_epochs,
            annealing_strategy=annealing_strategy,
            kl_start=kl_start,
        )
        # Training the smalltree subsplit
        for epoch in range(max_epochs):  # loop over the dataset multiple times
            train_one_epoch(
                train_small,
                self.module,
                optimizer,
                metrics_calc_train,
                epoch,
                device,
                train_small_tree=True,
                small_model=self.small_model,
                ind_leaf=ind_leaf,
            )
            if (epoch + 1) % 20 == 0:
                validate_one_epoch(
                    train_small,
                    self.module,
                    metrics_calc_val,
                    epoch,
                    device,
                    train_small_tree=True,
                    small_model=self.small_model,
                    ind_leaf=ind_leaf,
                )
            lr_scheduler.step()
            alpha_scheduler.on_epoch_end(epoch)

        return

    def train_tree(
        self,
        train_dataloader,
        train_dataloader_unshuffled,
        lr,
        weight_decay,
        num_epochs,
        annealing_strategy,
        kl_start,
        device,
        metrics_calc_train,
        metrics_calc_val,
    ):
        # Initialize optimizer and schedulers
        optimizer = get_optimizer(self.module, lr=lr, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        alpha_scheduler = AnnealKLCallback(
            self.module,
            epochs=num_epochs,
            annealing_strategy=annealing_strategy,
            kl_start=kl_start,
        )

        # finetune the full tree
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            train_one_epoch(
                train_dataloader,
                self.module,
                optimizer,
                metrics_calc_train,
                epoch,
                device,
            )
            if (epoch + 1) % 20 == 0:
                validate_one_epoch(
                    train_dataloader_unshuffled,
                    self.module,
                    metrics_calc_val,
                    epoch,
                    device,
                )
            lr_scheduler.step()
            alpha_scheduler.on_epoch_end(epoch)

    def stopping_criterion(
        self, train_dataloader_unshuffled, max_depth, batch_size, splitting_criterion
    ):
        (
            node_leaves_train,
            rec_loss_train,
            rec_loss_train_unweighted,
            elbo,
        ) = predict(
            train_dataloader_unshuffled,
            self.module,
            "node_leaves",
            "rec_loss_leafwise",
            "return_recloss_leafwise_unweighted",
            "elbo",
        )
        _, _, max_growth = compute_growing_leaf(
            train_dataloader_unshuffled,
            self.module,
            node_leaves_train,
            rec_loss_train,
            rec_loss_train_unweighted,
            elbo,
            max_depth,
            batch_size,
            max_leaves=self.n_cluster,
            check_max=True,
            splitting_criterion=splitting_criterion,
        )
        return max_growth
