"""
Utility functions for training.
"""

from typing import Literal

import torch
import math
import numpy as np
import wandb
from torch import Tensor
from tqdm import tqdm
import torch.optim as optim
from torchmetrics import Metric
from sklearn.metrics.cluster import normalized_mutual_info_score
from sctree.utils.utils import cluster_acc
from scvi._constants import REGISTRY_KEYS
from sklearn.decomposition import PCA
import torch
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import *
from sklearn import preprocessing


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    metrics_calc,
    epoch_idx,
    device,
    train_small_tree=False,
    small_model=None,
    ind_leaf=None,
):
    if train_small_tree:
        model.eval()
        small_model.train()
        model.return_bottomup[0] = True
        model.return_x[0] = True
        model.return_elbo[0] = True
        alpha = small_model.alpha
    else:
        model.train()
        alpha = model.alpha

    metrics_calc.reset()

    for batch_idx, batch in enumerate(tqdm(train_loader, leave=False)):

        labels = batch.pop("labels").to(device)
        batches = batch.get(REGISTRY_KEYS.BATCH_KEY, None)

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        if train_small_tree:
            # Gradient-free pass of full tree
            with torch.no_grad():
                outputs_full = model(batch, labels)
            # Why pass X here?
            x, node_leaves, bottom_up = (
                outputs_full["input"],
                outputs_full["node_leaves"],
                outputs_full["bottom_up"],
            )
            # Passing through subtree for updating its parameters
            if model.batch_correction == "diva":
                outputs = small_model(
                    batch,
                    node_leaves[ind_leaf]["z_sample"],
                    node_leaves[ind_leaf]["prob"],
                    bottom_up,
                    batch_embedding=outputs_full["batch_embedding"],
                )
            else:
                outputs = small_model(
                    batch,
                    node_leaves[ind_leaf]["z_sample"],
                    node_leaves[ind_leaf]["prob"],
                    bottom_up,
                )
            outputs["kl_root"] = torch.tensor(0.0, device=device)
        else:
            outputs = model.loss(batch, labels)
        # Compute the loss and its gradients
        rec_loss = outputs["rec_loss"]
        kl_losses = outputs["kl_root"] + outputs["kl_decisions"] + outputs["kl_nodes"]

        if model.batch_correction == "diva" and not train_small_tree:
            kl_losses += outputs["kl_batch"]
            batch_pred = outputs["ce_batch"]
        else:
            batch_pred = torch.tensor(0, device=device)

        loss_value = rec_loss + alpha * kl_losses + batch_pred
        loss_value.backward()

        # Adjust learning weights
        optimizer.step()

        # Store metrics
        y_pred = outputs["p_c_z"].argmax(
            dim=-1
        )  # Note that this is used for nmi, which means during subtreetraining, the nmi is calculate relative to only the subtree
        metrics_calc.update(
            loss_value,
            outputs["rec_loss"],
            outputs["kl_decisions"],
            outputs["kl_nodes"],
            outputs["kl_root"],
            (
                1 - torch.mean(y_pred.float())
                if outputs["p_c_z"].shape[1] <= 2
                else torch.tensor(0.0, device=device)
            ),
            labels,
            y_pred,
            batches,
            batch_pred,
        )

    if train_small_tree:
        model.return_bottomup[0] = False
        model.return_x[0] = False
        model.return_elbo[0] = False

    # Calculate and log metrics
    metrics = metrics_calc.compute()
    wandb.log({f"train/{k}": v for k, v in metrics.items()})
    # wandb.log({'train': metrics})
    prints = f"Epoch {epoch_idx}, Train     : "
    for key, value in metrics.items():
        prints += f"{key}: {value:.3f} "
    prints += f"kl_weight: {alpha:.3f}"
    print(prints)
    return


def validate_one_epoch(
    test_loader,
    model,
    metrics_calc,
    epoch_idx,
    device,
    train_small_tree=False,
    small_model=None,
    ind_leaf=None,
):
    model.eval()

    if train_small_tree:
        small_model.eval()
        model.return_bottomup[0] = True
        model.return_x[0] = True
        model.return_elbo[0] = True
        alpha = small_model.alpha
    else:
        alpha = model.alpha

    metrics_calc.reset()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, leave=False)):

            labels = batch.pop("labels").to(device)
            batches = batch.get(REGISTRY_KEYS.BATCH_KEY, None)
            # Make predictions for this batch
            if train_small_tree:
                # Sass of full tree
                outputs_full = model(batch, labels)
                x, node_leaves, bottom_up = (
                    outputs_full["input"],
                    outputs_full["node_leaves"],
                    outputs_full["bottom_up"],
                )
                # Passing through subtree
                if model.batch_correction == "diva":
                    outputs = small_model(
                        batch,
                        node_leaves[ind_leaf]["z_sample"],
                        node_leaves[ind_leaf]["prob"],
                        bottom_up,
                        batch_embedding=outputs_full["batch_embedding"],
                    )
                else:
                    outputs = small_model(
                        batch,
                        node_leaves[ind_leaf]["z_sample"],
                        node_leaves[ind_leaf]["prob"],
                        bottom_up,
                    )
                outputs["kl_root"] = torch.tensor(0.0, device=device)
            else:
                outputs = model(batch, labels)

            # Compute the loss and its gradients
            rec_loss = outputs["rec_loss"]
            kl_losses = (
                outputs["kl_root"] + outputs["kl_decisions"] + outputs["kl_nodes"]
            )

            if model.batch_correction == "diva" and not train_small_tree:
                kl_losses += outputs["kl_batch"]
                batch_pred = outputs["ce_batch"]
            else:
                batch_pred = torch.tensor(0, device=device)

            loss_value = rec_loss + alpha * kl_losses + batch_pred

            # Store metrics
            y_pred = outputs["p_c_z"].argmax(dim=-1)
            metrics_calc.update(
                loss_value,
                outputs["rec_loss"],
                outputs["kl_decisions"],
                outputs["kl_nodes"],
                outputs["kl_root"],
                (
                    1 - torch.mean(outputs["p_c_z"].argmax(dim=-1).float())
                    if outputs["p_c_z"].shape[1] <= 2
                    else torch.tensor(0.0, device=device)
                ),
                labels,
                y_pred,
                batches,
                batch_pred,
            )

    if train_small_tree:
        model.return_bottomup[0] = False
        model.return_x[0] = False
        model.return_elbo[0] = False

    # Calculate and log metrics
    metrics = metrics_calc.compute()

    wandb.log({f"validation/{k}": v for k, v in metrics.items()})
    prints = f"Epoch {epoch_idx}, Validation: "
    for key, value in metrics.items():
        prints += f"{key}: {value:.3f} "
    prints += f"kl_weight: {alpha:.3f}"
    print(prints)
    return


def predict(loader, model, *return_flags):
    model.eval()

    if "bottom_up" in return_flags:
        model.return_bottomup[0] = True
    if "X_aug" in return_flags:
        model.return_x[0] = True
    if "elbo" in return_flags:
        model.return_elbo[0] = True
    if "rec_loss_leafwise" in return_flags:
        model.return_recloss_leafwise[0] = True
    if "return_recloss_leafwise_unweighted" in return_flags:
        model.return_recloss_leafwise_unweighted[0] = True

    results = {name: [] for name in return_flags}
    # Create a dictionary to map return flags to corresponding functions
    return_functions = {
        "node_leaves": lambda: move_to(outputs["node_leaves"], "cpu"),
        "bottom_up": lambda: move_to(outputs["bottom_up"], "cpu"),
        "prob_leaves": lambda: move_to(outputs["p_c_z"], "cpu"),
        "X_aug": lambda: move_to(outputs["input"], "cpu"),
        "y": lambda: labels,
        "elbo": lambda: move_to(outputs["elbo_samples"], "cpu"),
        "rec_loss_leafwise": lambda: move_to(outputs["rec_loss_leafwise"], "cpu"),
        "return_recloss_leafwise_unweighted": lambda: move_to(
            outputs["return_recloss_leafwise_unweighted"], "cpu"
        ),
        "rec_loss": lambda: move_to(outputs["rec_loss"], "cpu"),
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            labels = batch.pop("labels")
            # Make predictions for this batch
            outputs = model(batch, labels)

            for return_flag in return_flags:
                results[return_flag].append(return_functions[return_flag]())

    for return_flag in return_flags:
        if return_flag == "bottom_up":
            bottom_up = results[return_flag]
            results[return_flag] = [
                torch.cat([sublist[i] for sublist in bottom_up], dim=0)
                for i in range(len(bottom_up[0]))
            ]
        elif return_flag == "node_leaves":
            node_leaves_combined = []
            node_leaves = results[return_flag]
            for i in range(len(node_leaves[0])):
                node_leaves_combined.append(dict())
                for key in node_leaves[0][i].keys():
                    node_leaves_combined[i][key] = torch.cat(
                        [sublist[i][key] for sublist in node_leaves], dim=0
                    )
            results[return_flag] = node_leaves_combined
        else:
            results[return_flag] = torch.cat(results[return_flag], dim=0)

    if "bottom_up" in return_flags:
        model.return_bottomup[0] = False
    if "X_aug" in return_flags:
        model.return_x[0] = False
    if "elbo" in return_flags:
        model.return_elbo[0] = False
    if "rec_loss_leafwise" in return_flags:
        model.return_recloss_leafwise[0] = False
    if "return_recloss_leafwise_unweighted" in return_flags:
        model.return_recloss_leafwise_unweighted[0] = False

    if len(return_flags) == 1:
        return list(results.values())[0]
    else:
        return tuple(results.values())


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, tuple):
        res = tuple(tensor.to(device) for tensor in obj)
        return res
    else:
        raise TypeError("Invalid type for move_to")


class AnnealKLCallback:
    def __init__(
        self,
        model,
        epochs: int,
        annealing_strategy: Literal["linear", "const"],
        kl_start: float,
    ):
        if annealing_strategy not in ["linear", "const"]:
            raise NotImplementedError(
                f"Annealing strategy {annealing_strategy} is not implemented."
            )
        self.annealing_strategy = annealing_strategy
        if epochs > 0:
            self.slope = 1.0 / epochs
        self.model = model
        self.kl_start = kl_start
        self.model.alpha = kl_start

    def on_epoch_end(self, epoch):
        if self.annealing_strategy == "linear":
            self.model.alpha = torch.tensor((epoch + 2) * self.slope)


class Decay:
    def __init__(self, lr=0.001, drop=0.1, epochs_drop=50):
        self.lr = lr
        self.drop = drop
        self.epochs_drop = epochs_drop

    def learning_rate_scheduler(self, epoch):
        initial_lrate = self.lr
        drop = self.drop
        epochs_drop = self.epochs_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate


def calc_aug_loss(prob_parent, prob_router, augmentation_methods, emb_contr=[]):
    aug_decisions_loss = torch.zeros(1, device=prob_parent.device)
    prob_parent = prob_parent.detach()

    num_losses = len(augmentation_methods)
    if emb_contr == [] and "instancewise_first" in augmentation_methods:
        num_losses = num_losses - 1
    if emb_contr == [] and "instancewise_full" in augmentation_methods:
        num_losses = num_losses - 1
    if num_losses <= 0:
        # If only instancewise losses and we're in smalltree
        return aug_decisions_loss

    # Get router probabilities of X' and X''
    p1, p2 = prob_router[: len(prob_router) // 2], prob_router[len(prob_router) // 2 :]
    # Perform invariance regularization
    for aug_method in augmentation_methods:
        if aug_method == "InfoNCE":
            p1_normed = torch.nn.functional.normalize(
                torch.stack([p1, 1 - p1], 1), dim=1
            )
            p2_normed = torch.nn.functional.normalize(
                torch.stack([p2, 1 - p2], 1), dim=1
            )
            pair_sim = torch.exp(torch.sum(p1_normed * p2_normed, dim=1))
            p_normed = torch.cat([p1_normed, p2_normed], dim=0)
            matrix_sim = torch.exp(torch.matmul(p_normed, p_normed.t()))
            norm_factor = torch.sum(matrix_sim, dim=1) - torch.diag(matrix_sim)
            pair_sim = pair_sim.repeat(2)  # storing sim for X' and X''
            info_nce_sample = -torch.log(pair_sim / norm_factor)
            info_nce = torch.sum(prob_parent * info_nce_sample) / torch.sum(prob_parent)
            aug_decisions_loss += info_nce

        elif aug_method in ["instancewise_first", "instancewise_full"]:
            looplen = (
                len(emb_contr)
                if aug_method == "instancewise_full"
                else min(len(emb_contr), 1)
            )
            for i in range(looplen):
                temp_instance = 0.5
                emb = emb_contr[i]
                emb1, emb2 = emb[: len(emb) // 2], emb[len(emb) // 2 :]
                emb1_normed = torch.nn.functional.normalize(emb1, dim=1)
                emb2_normed = torch.nn.functional.normalize(emb2, dim=1)
                pair_sim = torch.exp(
                    torch.sum(emb1_normed * emb2_normed, dim=1) / temp_instance
                )
                emb_normed = torch.cat([emb1_normed, emb2_normed], dim=0)
                matrix_sim = torch.exp(
                    torch.matmul(emb_normed, emb_normed.t()) / temp_instance
                )
                norm_factor = torch.sum(matrix_sim, dim=1) - torch.diag(matrix_sim)
                pair_sim = pair_sim.repeat(2)  # storing sim for X' and X''
                info_nce_sample = -torch.log(pair_sim / norm_factor)
                info_nce = torch.mean(info_nce_sample)
                info_nce = info_nce / looplen
                aug_decisions_loss += info_nce

        else:
            raise NotImplementedError

    # Also take into account that for smalltree, instancewise losses are 0
    aug_decisions_loss = aug_decisions_loss / num_losses

    return aug_decisions_loss


def get_ind_small_tree(node_leaves, n_effective_leaves):
    prob = node_leaves["prob"]
    ind = np.where(prob >= min(1 / n_effective_leaves, 0.5))[
        0
    ]  # To circumvent problems with n_effective_leaves==1
    return ind


def compute_leaves(tree):
    list_nodes = [{"node": tree, "depth": 0}]
    nodes_leaves = []
    while len(list_nodes) != 0:
        current_node = list_nodes.pop(0)
        node, depth_level = current_node["node"], current_node["depth"]
        if node.router is not None:
            node_left, node_right = node.left, node.right
            list_nodes.append({"node": node_left, "depth": depth_level + 1})
            list_nodes.append({"node": node_right, "depth": depth_level + 1})
        elif node.router is None and node.decoder is None:
            # We are in an internal node with pruned leaves and thus only have one child
            node_left, node_right = node.left, node.right
            child = node_left if node_left is not None else node_right
            list_nodes.append({"node": child, "depth": depth_level + 1})
        else:
            nodes_leaves.append(current_node)
    return nodes_leaves


def growing_leaf_purity(y_small):
    # check whether the selected node contains more than one digit
    # Compute the leaf purity by number of majority class samples divided by number of samples
    digits, counts = np.unique(y_small, return_counts=True)
    max_count = np.max(counts)
    purity = max_count / len(y_small)
    return purity


def compute_growing_leaf(
    loader_unshuffled,
    model,
    node_leaves,
    rec_loss,
    rec_loss_train_unweighted,
    elbo,
    max_depth,
    batch_size,
    max_leaves,
    splitting_criterion="n_samples",
    check_max=False,
):
    # count effective number of leaves
    weights = [node_leaves[i]["prob"] for i in range(len(node_leaves))]
    weights_summed = [weights[i].sum() for i in range(len(weights))]

    map_leaf = torch.stack(weights, dim=1).argmax(1)
    id, counts = torch.unique(map_leaf, return_counts=True)
    eff_leaves = torch.zeros(len(node_leaves), dtype=torch.long)
    eff_leaves[id] = counts
    eff_leaves = eff_leaves > 20
    n_effective_leaves = eff_leaves.sum().item()
    # n_effective_leaves = len(np.where(weights_summed / np.sum(weights_summed) >= 0.01)[0])
    print("\nNumber of effective leaves: ", n_effective_leaves)

    leaves = compute_leaves(model.tree)
    n_samples = []
    # split_oracle = []
    y_train = loader_unshuffled.dataset.dataset.adata_manager.get_from_registry(
        REGISTRY_KEYS.LABELS_KEY
    )
    # Calculating ground-truth nodes-to-split for logging and model development
    for i in range(len(node_leaves)):
        depth, node = leaves[i]["depth"], leaves[i]["node"]
        ind = get_ind_small_tree(node_leaves[i], n_effective_leaves)
        y_train_small = y_train[ind]

        # ##### TODO logging only during development
        # if eff_leaves[i] == False:
        #     purity = None
        # else:
        #     purity = growing_leaf_purity(y_train_small)

        # split_oracle.append(purity)

        # printing distribution of ground-truth classes in leaves
        print(f"Leaf {i}: ", np.unique(y_train_small, return_counts=True))
        n_samples.append(len(y_train_small))

        # rec_loss = [output_train['rec_loss'][i] for i in range(len(output_train['rec_loss']))]

        # grow until reaching required n_effective_leaves
    if n_effective_leaves >= max_leaves:
        print("\nReached maximum number of leaves\n")
        return None, None, True

    elif check_max:
        return None, None, False

    elif splitting_criterion in (
        "grow_all",
        "grow_all_rec_diff",
        "grow_all_rec_diff_short",
    ):
        # Convert all values in array that are False to None
        eff_leaves = [True if i else None for i in eff_leaves]
        return eff_leaves, None, n_effective_leaves

    elif splitting_criterion == "weighted_reconstruction":
        split_values = [
            (rec_loss[:, i] * weights[i]).sum() / weights_summed[i]
            for i in range(len(weights))
        ]
        # Highest weighted reconstruction loss indicates splitting
        ind_leaves = np.argsort(np.array(split_values))
        ind_leaves = ind_leaves[::-1]
    elif splitting_criterion == "reconstruction":
        split_values = [rec_loss[:, i].mean() for i in range(len(weights))]
        # Highest reconstruction loss indicates splitting
        ind_leaves = np.argsort(np.array(split_values))
        ind_leaves = ind_leaves[::-1]
    elif splitting_criterion == "weights":
        split_values = weights_summed
        # Highest weight of samples indicates splitting
        ind_leaves = np.argsort(np.array(split_values))
        ind_leaves = ind_leaves[::-1]
    elif splitting_criterion == "n_samples":
        split_values = n_samples
        # Highest number of samples indicates splitting
        ind_leaves = np.argsort(np.array(split_values))
        ind_leaves = ind_leaves[::-1]
    elif splitting_criterion == "grow_elbo_reduction":
        ind_leaves = np.arange(len(node_leaves))
    else:
        raise NotImplementedError

    # # Highest number of samples indicates splitting
    # split_values = n_samples
    # ind_leaves = np.argsort(np.array(split_values))
    # ind_leaves = ind_leaves[::-1]

    print("Ranking of leaves to split: ", ind_leaves)
    for i in ind_leaves:
        if n_samples[i] < batch_size:
            wandb.log({"Skipped Split": 1})
            print("We don't split leaves with fewer samples than batch size")
            continue
        elif leaves[i]["depth"] == max_depth or not leaves[i]["node"].expand:
            leaves[i]["node"].expand = False
            print("\nReached maximum architecture\n")
            print("\n!!ATTENTION!! architecture is not deep enough\n")
            continue
        else:
            ind_leaf = i
            leaf = leaves[ind_leaf]
            print(f"\nSplitting leaf {ind_leaf}\n")
            # Lower purity is better
            # wandb.log({'Split purity (lower=better)': split_oracle[ind_leaf]})
            # purity_ranking = np.argsort(np.array(split_oracle))
            # Find the rank of the chosen leaf
            # rank = np.where(purity_ranking == ind_leaf)[0][0]
            # Best rank is 0, which is lowest purity
            # wandb.log({'Split rank (0=best)': rank})

            return ind_leaf, leaf, n_effective_leaves

    return None, None, n_effective_leaves


def compute_pruning_leaf(model, node_leaves_train, n_cluster):
    leaves = compute_leaves(model.tree)
    n_leaves = len(node_leaves_train)
    weights = [node_leaves_train[i]["prob"] for i in range(n_leaves)]

    # Assign each sample to a leaf by argmax(weights)
    max_indeces = np.array([np.argmax(col) for col in zip(*weights)])

    n_samples = []
    for i in range(n_leaves):
        print(f"Leaf {i}: ", sum(max_indeces == i), "samples")
        n_samples.append(sum(max_indeces == i))

    # Prune leaves until n_cluster leaves are left
    ind_leaf = np.argmin(n_samples)
    if n_cluster < n_leaves:
        leaf = leaves[ind_leaf]
        return ind_leaf, leaf
    else:
        return None, None


def get_optimizer(model, lr: float, weight_decay: float):
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def move_data(data: dict[str, Tensor], device):
    return {k: v.to(device, non_blocking=True) for k, v in data.items()}


class Custom_Metrics(Metric):
    def __init__(self, device):
        super().__init__()
        self.add_state(
            "loss_value", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum"
        )
        self.add_state(
            "rec_loss", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum"
        )
        self.add_state(
            "kl_root", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum"
        )
        self.add_state(
            "kl_decisions",
            default=torch.tensor(0.0, device=device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "kl_nodes", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum"
        )
        self.add_state(
            "perc_samples",
            default=torch.tensor(0.0, device=device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "y_true",
            default=torch.tensor([], dtype=torch.int8, device=device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "y_pred",
            default=torch.tensor([], dtype=torch.int8, device=device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "batches", default=torch.tensor([], dtype=torch.int8), dist_reduce_fx="sum"
        )  # Batches stay on the CPU.
        self.add_state(
            "batch_pred",
            default=torch.tensor(0.0, device=device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "n_samples",
            default=torch.tensor(0, dtype=torch.int, device=device),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        loss_value: torch.Tensor,
        rec_loss: torch.Tensor,
        kl_decisions: torch.Tensor,
        kl_nodes: torch.Tensor,
        kl_root: torch.Tensor,
        perc_samples: torch.Tensor,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        batches: torch.Tensor,
        batch_pred: torch.Tensor,
    ):
        y_true = y_true.squeeze()
        assert y_true.shape == y_pred.shape

        n_samples = y_true.numel()
        self.n_samples += n_samples
        self.loss_value += loss_value.item() * n_samples
        self.rec_loss += rec_loss.item() * n_samples
        self.kl_root += kl_root.item() * n_samples
        self.kl_decisions += kl_decisions.item() * n_samples
        self.kl_nodes += kl_nodes.item() * n_samples
        self.perc_samples += perc_samples.item() * n_samples
        self.y_true = torch.cat((self.y_true, y_true))
        self.y_pred = torch.cat((self.y_pred, y_pred))
        if batches is not None:
            self.batches = torch.cat((self.batches, batches.ravel()))
        self.batch_pred += batch_pred.item() * n_samples

    def compute(self):
        nmi = normalized_mutual_info_score(
            self.y_true.cpu().numpy(), self.y_pred.cpu().numpy()
        )

        acc = cluster_acc(
            self.y_true.cpu().numpy(), self.y_pred.cpu().numpy(), return_index=False
        )
        metrics = dict(
            {
                "loss_value": self.loss_value / self.n_samples,
                "rec_loss": self.rec_loss / self.n_samples,
                "kl_decisions": self.kl_decisions / self.n_samples,
                "kl_root": self.kl_root / self.n_samples,
                "kl_nodes": self.kl_nodes / self.n_samples,
                "perc_samples": self.perc_samples / self.n_samples,
                "ce_batch": self.batch_pred / self.n_samples,
                "nmi": nmi,
                "accuracy": acc,
            }
        )

        if len(self.batches) > 0:
            nmi_batch = 1 - normalized_mutual_info_score(
                self.batches.numpy(), self.y_pred.cpu().numpy()
            )
            metrics["nmi_batch"] = nmi_batch

        return metrics
