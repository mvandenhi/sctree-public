from typing import Literal, Optional

import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseMinifiedModeModuleClass
from torch import nn

from sctree.models.losses import loss_nb, loss_nb_leafwise, loss_nb_leafwise_unweighted
from sctree.models.networks import (
    get_encoder,
    MLP,
    Dense,
    Router,
    DecoderTree,
    LinearDecoderTree,
    BatchDecoder,
)
from sctree.utils.model_utils import construct_tree, compute_posterior, return_list_tree
import torch.distributions as td
import torch.nn.functional as F
from sctree.utils.training_utils import move_data


class SmallTreeModule(BaseMinifiedModeModuleClass):
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        depth: int,
        n_batch: int,
        n_cats_per_cov: int,
        encoded_size_gen: int,
        px_r,
        batch_decoder,
        likelihood,
        n_continuous_cov: int,
        n_hidden: int = 128,
        kl_start: float = 0.0,
        decoder_n_layers: int = 1,
        router_n_layers: int = 1,
        router_dropout: float = 0.1,
        mlp_n_layers: int = 1,
    ):
        super().__init__()
        # KL-annealing weight initialization
        self.alpha = kl_start

        self.depth = depth
        self.encoded_size = n_latent
        self.hidden_layer = n_hidden
        self.inp_shape = n_input
        self.denses = nn.ModuleList(
            [Dense(self.hidden_layer, self.encoded_size) for _ in range(2)]
        )
        self.transformations = nn.ModuleList(
            [
                MLP(
                    self.encoded_size,
                    self.encoded_size,
                    self.hidden_layer,
                    n_layers=mlp_n_layers,
                )
                for _ in range(2)
            ]
        )
        self.decision = Router(
            self.encoded_size,
            hidden_units=self.hidden_layer,
            n_layers=router_n_layers,
            dropout=router_dropout,
        )
        self.decision_q = Router(
            self.hidden_layer,
            hidden_units=self.hidden_layer,
            n_layers=router_n_layers,
            dropout=router_dropout,
        )
        self.likelihood = likelihood

        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        n_input_decoder = encoded_size_gen + n_continuous_cov
        if decoder_n_layers > 0:
            self.decoders = nn.ModuleList(
                [
                    DecoderTree(
                        n_input=n_input_decoder,
                        n_output=self.inp_shape,
                        px_r=px_r,
                        n_cat_list=cat_list,
                        batch_decoder=batch_decoder,
                        n_layers=decoder_n_layers,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.decoders = nn.ModuleList(
                [
                    LinearDecoderTree(
                        n_input=n_input_decoder,
                        n_output=self.inp_shape,
                        px_r=px_r,
                        n_cat_list=cat_list,
                        batch_decoder=batch_decoder,
                    )
                    for _ in range(2)
                ]
            )

    def forward(self, tensors, z_parent, p, bottom_up, batch_embedding=None):
        epsilon = 1e-7  # Small constant to prevent numerical instability
        tensors = move_data(tensors, self.device)
        x_org = tensors[REGISTRY_KEYS.X_KEY]
        x = x_org
        library = torch.log(x.sum(1)).unsqueeze(dim=1)
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        # Extract relevant bottom-up
        d_q = bottom_up[-self.depth]
        d = bottom_up[-self.depth - 1]

        prob_child_left = self.decision(z_parent).squeeze()
        prob_child_left_q = self.decision_q(d_q).squeeze()
        leaves_prob = [p * prob_child_left_q, p * (1 - prob_child_left_q)]

        kl_decisions = prob_child_left_q * torch.log(
            epsilon + prob_child_left_q / (prob_child_left + epsilon)
        ) + (1 - prob_child_left_q) * torch.log(
            epsilon + (1 - prob_child_left_q) / (1 - prob_child_left + epsilon)
        )
        kl_decisions = torch.mean(p * kl_decisions)

        reconstructions = []
        kl_nodes = torch.zeros(1, device=self.device)
        for i in range(2):
            # Compute posterior parameters
            z_mu_q_hat, z_sigma_q_hat = self.denses[i](d)
            _, z_mu_p, z_sigma_p = self.transformations[i](z_parent)
            z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 1)
            z_mu_q, z_sigma_q = compute_posterior(
                z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p
            )

            # Compute sample z using mu_q and sigma_q
            z_q = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
            z_sample = z_q.rsample()

            # Compute KL node
            kl_node = torch.mean(leaves_prob[i] * td.kl_divergence(z_q, z_p))
            kl_nodes += kl_node

            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            if cont_covs is None:
                decoder_input = z_sample
            elif z_sample.dim() != cont_covs.dim():
                decoder_input = torch.cat(
                    [z_sample, cont_covs.unsqueeze(0).expand(z_sample.size(0), -1, -1)],
                    dim=-1,
                )
            else:
                decoder_input = torch.cat([z_sample, cont_covs], dim=-1)

            if cat_covs is not None:
                categorical_input = torch.split(cat_covs, 1, dim=1)
            else:
                categorical_input = ()

            if batch_embedding is not None:
                decoder_input = torch.cat([decoder_input, batch_embedding], dim=-1)
            reconstructions.append(
                self.decoders[i](
                    "gene", decoder_input, library, batch_index, *categorical_input
                )
            )

        kl_nodes_loss = torch.clamp(kl_nodes, min=-10, max=1e10)

        # Probability of falling in each leaf
        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)

        rec_losses = loss_nb(
            x_org, reconstructions, leaves_prob, likelihood=self.likelihood
        )
        rec_loss = torch.mean(rec_losses, dim=0)

        return {
            "rec_loss": rec_loss,
            "weights": leaves_prob,
            "kl_decisions": kl_decisions,
            "kl_nodes": kl_nodes_loss,
            "p_c_z": p_c_z,
        }

    def freeze_dispersion(self):
        for decoder in self.decoders:
            decoder.px_r.requires_grad = False

    def unfreeze_dispersion(self):
        for decoder in self.decoders:
            decoder.px_r.requires_grad = True


class TreeModule(BaseMinifiedModeModuleClass):
    def __init__(
        self,
        n_input: int,
        inital_depth: int,
        n_continuous_cov: int,
        n_batch: int,
        n_cats_per_cov: int,
        likelihood: Literal["NB", "ZINB"],
        max_depth: int = 5,
        n_latent: int = 2,
        n_hidden: int = 128,
        kl_start: float = 0.0,
        lambda_diva_kl: float = 1.0,
        lambda_diva_ce: float = 1.0,
        encoder_arch: str = "mlp",
        decoder_n_layers: int = 1,
        router_n_layers: int = 1,
        router_dropout: float = 0.1,
        mlp_n_layers: int = 1,
        dispersion: Literal["gene", "cluster"] = "gene",
        batch_correction: Literal["global", "cluster", "diva"] = "global",
    ):
        super().__init__()
        # saving important variables to initialize the tree
        self.encoded_sizes = [n_latent] * (max_depth + 1)
        self.hidden_layers = [n_hidden] * (max_depth + 1)
        self.alpha = kl_start
        self.likelihood = likelihood
        self.batch_correction = batch_correction
        self.lambda_diva_kl = lambda_diva_kl
        self.lambda_diva_ce = lambda_diva_ce

        # check that the number of layers for bottom up is equal to top down
        self.depth = inital_depth
        self.inp_shape = n_input
        self.encoder_arch = encoder_arch

        self.return_x = torch.tensor([False])
        self.return_bottomup = torch.tensor([False])
        self.return_elbo = torch.tensor([False])
        self.return_recloss_leafwise = torch.tensor([False])
        self.return_recloss_leafwise_unweighted = torch.tensor([False])

        # bottom up: the inference chain that from x computes the d units till the root
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder = get_encoder(
            architecture=encoder_arch,
            encoded_size=self.hidden_layers[0],
            x_shape=self.inp_shape,
            n_cat_list=cat_list,
        )

        self.bottom_up = nn.ModuleList([encoder])
        for i in range(1, len(self.hidden_layers)):
            self.bottom_up.append(
                MLP(
                    self.hidden_layers[i - 1],
                    self.encoded_sizes[i],
                    self.hidden_layers[i],
                    n_layers=mlp_n_layers,
                )
            )

        # top down: the generative model that from x computes the prior prob of all nodes from root till leaves
        # it has a tree structure which is constructed by passing a list of transformations and routers from root to
        # leaves visiting nodes layer-wise from left to right
        # N.B. root has None as transformation and leaves have None as routers
        # the encoded sizes and layers are reversed from bottom up

        # select the top down generative networks
        encoded_size_gen = self.encoded_sizes[
            -(self.depth + 1) :
        ]  # e.g. encoded_sizes 32,16,8, depth 1
        layers_gen = self.hidden_layers[
            -(self.depth + 1) :
        ]  # e.g. encoded_sizes 256,128,64, depth 1

        # add root transformation and dense layer, the dense layer is layer that connects the bottom-up with the nodes
        self.transformations = nn.ModuleList([None])
        self.denses = nn.ModuleList([Dense(layers_gen[0], encoded_size_gen[0])])
        for i in range(self.depth):
            for j in range(2 ** (i + 1)):
                self.transformations.append(
                    MLP(
                        encoded_size_gen[i],
                        encoded_size_gen[i + 1],
                        layers_gen[i],
                        n_layers=mlp_n_layers,
                    )
                )  # MLP from depth i to i+1
                self.denses.append(
                    Dense(layers_gen[i + 1], encoded_size_gen[i + 1])
                )  # Dense at depth i+1 from bottom-up to top-down

        self.decisions = nn.ModuleList([])
        for i in range(self.depth):
            for j in range(2**i):
                self.decisions.append(
                    Router(
                        encoded_size_gen[i],
                        hidden_units=layers_gen[i],
                        n_layers=router_n_layers,
                        dropout=router_dropout,
                    )
                )  # Router at node of depth i

        # decoders = [None, None, None, Dec, Dec, Dec, Dec]
        self.decoders = nn.ModuleList(
            [None for i in range(self.depth) for j in range(2**i)]
        )
        # the leaves do not have decisions but have decoders

        if dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(self.inp_shape))
        else:
            self.px_r = None

        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        n_input_decoder = encoded_size_gen[-1] + n_continuous_cov

        if self.batch_correction == "global":
            batch_n_out = (
                self.inp_shape if decoder_n_layers == 0 else self.hidden_layers[0]
            )
            self.batch_decoder = BatchDecoder(n_cat_list=cat_list, n_out=batch_n_out)
        elif self.batch_correction == "cluster":
            self.batch_decoder = None
        elif self.batch_correction == "diva":
            self.n_batch = n_batch
            if self.n_batch == 1:
                dim_batch_space = 0
            else:
                dim_batch_space = 5
                self.batch_encoder = MLP(
                    self.inp_shape, dim_batch_space, encoded_size_gen[0], n_layers=1
                )
                self.batch_classifier = nn.Linear(dim_batch_space, self.n_batch)
            self.batch_decoder = dim_batch_space
        else:
            raise NotImplementedError(f"{self.batch_correction} is not implemented.")

        for _ in range(2 ** (self.depth)):
            self.decisions.append(None)
            if decoder_n_layers > 0:
                self.decoders.append(
                    DecoderTree(
                        n_input=n_input_decoder,
                        n_output=self.inp_shape,
                        n_hidden=n_hidden,
                        px_r=self.px_r,
                        n_cat_list=cat_list,
                        n_layers=decoder_n_layers,
                        batch_decoder=self.batch_decoder,
                    )
                )
            else:
                self.decoders.append(
                    LinearDecoderTree(
                        n_input=n_input_decoder,
                        n_output=self.inp_shape,
                        px_r=self.px_r,
                        n_cat_list=cat_list,
                        batch_decoder=self.batch_decoder,
                    )
                )

        # bottom-up decisions
        self.decisions_q = nn.ModuleList([])
        for i in range(self.depth):
            for _ in range(2**i):
                self.decisions_q.append(
                    Router(
                        layers_gen[i],
                        hidden_units=layers_gen[i],
                        n_layers=router_n_layers,
                        dropout=router_dropout,
                    )
                )
        for _ in range(2 ** (self.depth)):
            self.decisions_q.append(None)

        # set seed according to time
        import time

        torch.manual_seed(int(time.time()))
        self.random_label = torch.randint(0, 3, (1,))
        # construct the tree
        self.tree = construct_tree(
            transformations=self.transformations,
            routers=self.decisions,
            routers_q=self.decisions_q,
            denses=self.denses,
            decoders=self.decoders,
        )

    def forward(
        self,
        tensors,
        labels,
        get_inference_input_kwargs=None,
        get_generative_input_kwargs=None,
        inference_kwargs=None,
        generative_kwargs=None,
        loss_kwargs=None,
        compute_loss=True,
    ):
        return self.loss(tensors, labels)

    def loss(self, tensors, labels=None):

        tensors = move_data(tensors, self.device)

        epsilon = 1e-7
        x_org = tensors[REGISTRY_KEYS.X_KEY]
        x = x_org
        library = torch.log(x.sum(1)).unsqueeze(dim=1)
        x = torch.log(x + 1.0)
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        # compute deterministic bottom up
        d = x
        encoders = []

        if "scvi" in self.encoder_arch:
            d, _, _ = self.bottom_up[0](d, batch_index, *categorical_input)

        else:
            d, _, _ = self.bottom_up[0](d)
        encoders.append(d)
        for i in range(1, len(self.hidden_layers)):
            d, _, _ = self.bottom_up[i](d)
            # store bottom-up embeddings for top-down
            encoders.append(d)

        # compute DIVA latent space
        if self.batch_correction == "diva":
            if self.n_batch == 1:
                kl_batch = torch.tensor(0, device=self.device)
                ce_batch = torch.tensor(0, device=self.device)
                batch_embedding = None
            else:
                _, batch_embedding_mu, batch_embedding_sigma = self.batch_encoder(x)
                batch_embedding_dist = td.Independent(
                    td.Normal(
                        batch_embedding_mu, torch.sqrt(batch_embedding_sigma + epsilon)
                    ),
                    1,
                )
                batch_p = td.Independent(
                    td.Normal(
                        torch.zeros_like(batch_embedding_mu),
                        torch.ones_like(batch_embedding_sigma),
                    ),
                    1,
                )
                batch_embedding = batch_embedding_dist.rsample()
                batch_logits = self.batch_classifier(batch_embedding)

                # Compute losses
                kl_batch = self.lambda_diva_kl * td.kl_divergence(
                    batch_embedding_dist, batch_p
                ).mean(0)
                assert batch_index.dim() == 2 and batch_index.shape[1] == 1
                ce_batch = self.lambda_diva_ce * F.cross_entropy(
                    batch_logits, batch_index[:, 0].long()
                )

        # create a list of nodes of the tree that need to be processed
        list_nodes = [
            {
                "node": self.tree,
                "depth": 0,
                "prob": torch.ones(x.size(0), device=self.device),
                "z_parent_sample": None,
            }
        ]
        # initializate KL losses
        kl_nodes_tot = torch.zeros(len(x), device=self.device)
        kl_decisions_tot = torch.zeros(len(x), device=self.device)
        leaves_prob = []
        reconstructions = []
        node_leaves = []
        while len(list_nodes) != 0:
            # store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level, prob = (
                current_node["node"],
                current_node["depth"],
                current_node["prob"],
            )
            z_parent_sample = current_node["z_parent_sample"]
            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[-(1 + depth_level)]
            z_mu_q_hat, z_sigma_q_hat = node.dense(d)

            if depth_level == 0:
                # here we are in the root
                # standard gaussian
                z_mu_p, z_sigma_p = torch.zeros_like(z_mu_q_hat), torch.ones_like(
                    z_sigma_q_hat
                )
                z_p = td.Independent(
                    td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 1
                )
                # sampled z is the top layer of deterministic bottom-up
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
            else:
                # the generative mu and sigma is the output of the top-down network given the sampled parent
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                z_p = td.Independent(
                    td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 1
                )
                z_mu_q, z_sigma_q = compute_posterior(
                    z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p
                )

            # compute sample z using mu_q and sigma_q
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
            z_sample = z.rsample()

            # compute KL node
            kl_node = prob * td.kl_divergence(z, z_p)
            kl_node = torch.clamp(kl_node, min=-1, max=1000)

            if depth_level == 0:
                kl_root = kl_node
            else:
                kl_nodes_tot += kl_node

            if node.router is not None:
                # we are in the internal nodes (not leaves)
                prob_child_left = node.router(z_sample).squeeze()
                prob_child_left_q = node.routers_q(d).squeeze()

                kl_decisions = (
                    prob_child_left_q
                    * (epsilon + prob_child_left_q / (prob_child_left + epsilon)).log()
                    + (1 - prob_child_left_q)
                    * (
                        epsilon
                        + (1 - prob_child_left_q) / (1 - prob_child_left + epsilon)
                    ).log()
                )
                kl_decisions = prob * kl_decisions
                kl_decisions_tot += kl_decisions

                # we are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (
                    1 - prob_child_left_q
                )

                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {
                        "node": node_left,
                        "depth": depth_level + 1,
                        "prob": prob_node_left,
                        "z_parent_sample": z_sample,
                    }
                )
                list_nodes.append(
                    {
                        "node": node_right,
                        "depth": depth_level + 1,
                        "prob": prob_node_right,
                        "z_parent_sample": z_sample,
                    }
                )
            elif node.decoder is not None:
                # if we are in a leaf we need to store the prob of reaching that leaf and compute reconstructions
                # as the nodes are explored left to right, these probabilities will be also ordered left to right
                leaves_prob.append(prob)

                if cont_covs is None:
                    decoder_input = z_sample
                elif z_sample.dim() != cont_covs.dim():
                    decoder_input = torch.cat(
                        [
                            z_sample,
                            cont_covs.unsqueeze(0).expand(z_sample.size(0), -1, -1),
                        ],
                        dim=-1,
                    )
                else:
                    decoder_input = torch.cat([z_sample, cont_covs], dim=-1)

                if cat_covs is not None:
                    categorical_input = torch.split(cat_covs, 1, dim=1)
                else:
                    categorical_input = ()

                if self.batch_correction == "diva":
                    if self.n_batch != 1:
                        decoder_input = torch.cat(
                            [decoder_input, batch_embedding], dim=-1
                        )

                dec = node.decoder
                reconstructions.append(
                    dec("gene", decoder_input, library, batch_index, *categorical_input)
                )
                node_leaves.append({"prob": prob, "z_sample": z_sample})

            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {
                        "node": child,
                        "depth": depth_level + 1,
                        "prob": prob,
                        "z_parent_sample": z_sample,
                    }
                )

        kl_nodes_loss = torch.clamp(torch.mean(kl_nodes_tot), min=-10, max=1e10)
        kl_decisions_loss = torch.mean(kl_decisions_tot)
        kl_root_loss = torch.mean(kl_root)

        # p_c_z is the probability of reaching a leaf and should be of shape [batch_size, num_clusters]
        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)

        rec_losses = loss_nb(
            x_org, reconstructions, leaves_prob, likelihood=self.likelihood
        )
        rec_loss = torch.mean(rec_losses, dim=0)

        return_dict = {
            "rec_loss": rec_loss,
            "weights": leaves_prob,
            "kl_root": kl_root_loss,
            "kl_decisions": kl_decisions_loss,
            "kl_nodes": kl_nodes_loss,
            "p_c_z": p_c_z,
            "node_leaves": node_leaves,
        }

        if self.return_elbo:
            return_dict["elbo_samples"] = (
                kl_nodes_tot + kl_decisions_tot + kl_root + rec_losses
            )

        if self.return_bottomup:
            return_dict["bottom_up"] = encoders

        if self.return_x:
            return_dict["input"] = x

        if self.return_recloss_leafwise:
            return_dict["rec_loss_leafwise"] = loss_nb_leafwise(
                x_org, reconstructions, leaves_prob, likelihood=self.likelihood
            )

        if self.return_recloss_leafwise_unweighted:
            return_dict["return_recloss_leafwise_unweighted"] = (
                loss_nb_leafwise_unweighted(
                    x_org, reconstructions, leaves_prob, likelihood=self.likelihood
                )
            )

        if self.batch_correction == "diva":
            return_dict["kl_batch"] = kl_batch
            return_dict["ce_batch"] = ce_batch
            return_dict["batch_embedding"] = batch_embedding

        return return_dict

    def compute_leaves(self):
        # returns leaves of the tree
        list_nodes = [{"node": self.tree, "depth": 0}]
        nodes_leaves = []
        while len(list_nodes) != 0:
            current_node = list_nodes.pop(0)
            node, depth_level = current_node["node"], current_node["depth"]
            if node.router is not None:
                node_left, node_right = node.left, node.right
                list_nodes.append({"node": node_left, "depth": depth_level + 1})
                list_nodes.append({"node": node_right, "depth": depth_level + 1})
            elif node.router is None and node.decoder is None:
                # we are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append({"node": child, "depth": depth_level + 1})
            else:
                nodes_leaves.append(current_node)
        return nodes_leaves

    def compute_depth(self):
        # computes depth of the tree
        nodes_leaves = self.compute_leaves()
        d = []
        for i in range(len(nodes_leaves)):
            d.append(nodes_leaves[i]["depth"])
        return max(d)

    def attach_smalltree(self, node, small_model: SmallTreeModule):
        # attaching a (trained) smalltree to the full tree
        assert node.left is None and node.right is None
        node.router = small_model.decision
        node.routers_q = small_model.decision_q
        node.decoder = None
        small_model.unfreeze_dispersion()
        for j in range(2):
            dense = small_model.denses[j]
            transformation = small_model.transformations[j]
            decoder = small_model.decoders[j]
            node.insert(transformation, None, None, dense, decoder)

        transformations, routers, denses, decoders, routers_q = return_list_tree(
            self.tree
        )

        self.decisions_q = routers_q
        self.transformations = transformations
        self.decisions = routers
        self.denses = denses
        self.decoders = decoders
        self.depth = self.compute_depth()

    def compute_reconstruction(self, x):
        assert self.training is False
        epsilon = 1e-7
        device = x.device

        # compute deterministic bottom up
        d = x
        encoders = []

        for i in range(0, len(self.hidden_layers)):
            d, _, _ = self.bottom_up[i](d)
            # store the bottom-up layers for the top down computation
            encoders.append(d)

        # create a list of nodes of the tree that need to be processed
        list_nodes = [
            {
                "node": self.tree,
                "depth": 0,
                "prob": torch.ones(x.size(0), device=device),
                "z_parent_sample": None,
            }
        ]

        # initializate KL losses
        leaves_prob = []
        reconstructions = []
        node_leaves = []
        while len(list_nodes) != 0:

            # store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level, prob = (
                current_node["node"],
                current_node["depth"],
                current_node["prob"],
            )
            z_parent_sample = current_node["z_parent_sample"]
            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[-(1 + depth_level)]
            z_mu_q_hat, z_sigma_q_hat = node.dense(d)

            if depth_level == 0:
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
            else:
                # the generative mu and sigma is the output of the top-down network given the sampled parent
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                z_mu_q, z_sigma_q = compute_posterior(
                    z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p
                )

            # compute sample z using mu_q and sigma_q
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
            z_sample = z.rsample()

            # if we are in the internal nodes (not leaves)
            if node.router is not None:

                prob_child_left_q = node.routers_q(d).squeeze()

                # we are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (
                    1 - prob_child_left_q
                )

                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {
                        "node": node_left,
                        "depth": depth_level + 1,
                        "prob": prob_node_left,
                        "z_parent_sample": z_sample,
                    }
                )
                list_nodes.append(
                    {
                        "node": node_right,
                        "depth": depth_level + 1,
                        "prob": prob_node_right,
                        "z_parent_sample": z_sample,
                    }
                )
            elif node.decoder is not None:
                # if we are in a leaf we need to store the prob of reaching that leaf and compute reconstructions
                # as the nodes are explored left to right, these probabilities will be also ordered left to right
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))
                node_leaves.append({"prob": prob, "z_sample": z_sample})

            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {
                        "node": child,
                        "depth": depth_level + 1,
                        "prob": prob,
                        "z_parent_sample": z_sample,
                    }
                )

        return reconstructions, node_leaves
