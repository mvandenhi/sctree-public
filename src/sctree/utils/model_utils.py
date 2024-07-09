"""
Utility functions for model.
"""
import numpy as np
import torch.nn as nn

def compute_posterior(mu_q, mu_p, sigma_q, sigma_p):
    epsilon = 1e-7 
    z_sigma_q = 1 / (1 / (sigma_q + epsilon) + 1 / (sigma_p + epsilon))
    z_mu_q = (mu_q / (sigma_q + epsilon) +
              mu_p / (sigma_p + epsilon)) * z_sigma_q
    return z_mu_q, z_sigma_q


def construct_tree(transformations, routers, routers_q, denses, decoders):
    """
        Construct the tree by passing a list of transformations and routers from root to leaves visiting nodes
        layer-wise from left to right

        :param transformations: list of transformations to attach to the nodes of the tree
        :param routers: list of decisions to attach to the nodes of the tree
        :param denses: list of dense network that from d of the bottom up compute node-specific q
        :param decoders: list of decoders to attach to the nodes, they should be set to None except the leaves
        :return:the root of the tree
        """
    if len(transformations) != len(routers) and len(transformations) != len(denses) \
            and len(transformations) != len(decoders):
        raise ValueError('Len transformation is different than len routers in constructing the tree.')
    root = Node(transformation=transformations[0], router=routers[0], routers_q=routers_q[0], dense=denses[0], decoder=decoders[0])
    for i in range(1, len(transformations)):
        root.insert(transformation=transformations[i], router=routers[i], routers_q=routers_q[i], dense=denses[i], decoder=decoders[i])
    return root


class Node:
    def __init__(self, transformation, router, routers_q, dense, decoder=None, expand=True):
        self.left = None
        self.right = None
        self.parent = None
        self.transformation = transformation
        self.dense = dense
        self.router = router
        self.routers_q = routers_q
        self.decoder = decoder
        self.expand = expand

    def insert(self, transformation=None, router=None, routers_q=None, dense=None, decoder=None):
        queue = []
        node = self
        queue.append(node)
        while len(queue) > 0:
            node = queue.pop(0)
            if node.expand:
                if node.left is None:
                    node.left = Node(transformation, router, routers_q, dense, decoder)
                    node.left.parent = node
                    return
                elif node.right is None:
                    node.right = Node(transformation, router, routers_q, dense, decoder)
                    node.right.parent = node
                    return
                else:
                    queue.append(node.left)
                    queue.append(node.right)
        print('\nAttention node has not been inserted!\n')
        return

    def prune_child(self, child):
        if child is self.left:
            self.left = None
            self.router = None

        elif child is self.right:
            self.right = None
            self.router = None

        else:
            raise ValueError("This is not my child! (Node is not a child of this parent.)")

def return_list_tree(root):
    list_nodes = [root]
    denses = []
    transformations = []
    routers = []
    routers_q = []
    decoders = []
    while len(list_nodes) != 0:
        current_node = list_nodes.pop(0)
        denses.append(current_node.dense)
        transformations.append(current_node.transformation)
        routers.append(current_node.router)
        routers_q.append(current_node.routers_q)
        decoders.append(current_node.decoder)
        if current_node.router is not None:
            node_left, node_right = current_node.left, current_node.right
            list_nodes.append(node_left)
            list_nodes.append(node_right)
        elif current_node.router is None and current_node.decoder is None:
            # We are in an internal node with pruned leaves and thus only have one child
            node_left, node_right = current_node.left, current_node.right
            child = node_left if node_left is not None else node_right
            list_nodes.append(child)
    return nn.ModuleList(transformations), nn.ModuleList(routers), nn.ModuleList(denses), nn.ModuleList(decoders), nn.ModuleList(routers_q)


def construct_tree_fromnpy(model, data_tree, configs):
    from sctree.module.treemodule import SmallTreeModule
    nodes = {0: {'node': model.module.tree, 'depth': 0}}

    for i in range(1, len(data_tree)-1):
        node_left = data_tree[i]
        node_right = data_tree[i + 1]
        id_node_left = node_left[0]
        id_node_right = node_right[0]

        if node_left[2] == node_right[2]:
            id_parent = node_left[2]

            parent = nodes[id_parent]
            node = parent['node']
            depth = parent['depth']

            new_depth = depth + 1

            small_model = SmallTreeModule(depth=new_depth, n_input=model.summary_stats.n_vars,
                                               n_latent=model.n_latent,
                                               px_r=model.module.px_r,
                                               n_batch=model.summary_stats.n_batch, encoded_size_gen=model.n_latent,
                                               n_hidden=model.n_hidden,
                                               n_continuous_cov=model.summary_stats.get("n_extra_continuous_covs", 0),
                                               n_cats_per_cov=model.n_cats_per_cov, kl_start=model.kl_start,
                                               decoder_n_layers=model.decoder_n_layers, 
                                               router_n_layers=model.router_n_layers, router_dropout=model.router_dropout,
                                               mlp_n_layers=model.mlp_n_layers,
                                               likelihood=model.likelihood)
            node.router = small_model.decision
            node.routers_q = small_model.decision_q

            node.decoder = None
            n = []
            for j in range(2):
                dense = small_model.denses[j]
                transformation = small_model.transformations[j]
                decoder = small_model.decoders[j]
                n.append(Node(transformation, None, None, dense, decoder))

            node.left = n[0]
            node.right = n[1]

            nodes[id_node_left] = {'node': node.left, 'depth': new_depth}
            nodes[id_node_right] = {'node': node.right, 'depth': new_depth}
        elif data_tree[i][2] != data_tree[i - 1][2]: # Internal node w/ 1 child only
            id_parent = node_left[2]

            parent = nodes[id_parent]
            node = parent['node']
            depth = parent['depth']

            new_depth = depth + 1

            small_model = SmallTreeModule(depth=new_depth, n_input=model.summary_stats.n_vars,
                                               n_latent=model.n_latent,
                                               px_r=model.module.px_r,
                                               n_batch=model.summary_stats.n_batch, encoded_size_gen=model.n_latent,
                                               n_hidden=model.n_hidden,
                                               n_continuous_cov=model.summary_stats.get("n_extra_continuous_covs", 0),
                                               n_cats_per_cov=model.n_cats_per_cov, kl_start=model.kl_start,
                                               decoder_n_layers=model.decoder_n_layers, 
                                               router_n_layers=model.router_n_layers, router_dropout=model.router_dropout,
                                               mlp_n_layers=model.mlp_n_layers,
                                               likelihood=model.likelihood)

            node.router = None
            node.routers_q = None

            node.decoder = None
            n = []
            for j in range(1):
                dense = small_model.denses[j]
                transformation = small_model.transformations[j]
                decoder = small_model.decoders[j]
                n.append(Node(transformation, None, None, dense, decoder))

            node.left = n[0]

            nodes[id_node_left] = {'node': node.left, 'depth': new_depth}
            
            

    transformations, routers, denses, decoders, routers_q = return_list_tree(model.module.tree)
    model.module.decisions_q = routers_q
    model.module.transformations = transformations
    model.module.decisions = routers
    model.module.denses = denses
    model.module.decoders = decoders
    model.module.depth = model.module.compute_depth()
    return model



def construct_data_tree_wandb(model, y_predicted, y_true, n_leaves):
    list_nodes = [{'node':model.tree, 'id': 0, 'parent':None}]
    data = []
    i = 0
    labels = [i for i in range(n_leaves)]
    while len(list_nodes) != 0:
        current_node = list_nodes.pop(0)
        if current_node['node'].router is not None:
            data.append([current_node['id'], str(current_node['id']), current_node['parent'], 10])
            node_left, node_right = current_node['node'].left, current_node['node'].right
            i += 1
            list_nodes.append({'node':node_left, 'id': i, 'parent': current_node['id']})
            i += 1
            list_nodes.append({'node':node_right, 'id': i, 'parent': current_node['id']})
        elif current_node['node'].router is None and current_node['node'].decoder is None:
            # We are in an internal node with pruned leaves and will only add the non-pruned leaves
            data.append([current_node['id'], str(current_node['id']), current_node['parent'], 10])
            node_left, node_right = current_node['node'].left, current_node['node'].right
            child = node_left if node_left is not None else node_right
            i += 1
            list_nodes.append({'node': child, 'id': i, 'parent': current_node['id']})
        else:
            y_leaf = labels.pop(0)
            ind = np.where(y_predicted == y_leaf)[0]
            digits, counts = np.unique(y_true[ind], return_counts=True)
            tot = len(ind)
            if tot == 0:
                name = 'no digits'
            else:
                counts = np.round(counts / np.sum(counts), 2)
                ind = np.where(counts > 0.1)[0]
                name = ' '
                for j in ind:
                    name = name + str(digits[j]) + ': ' + str(counts[j]) + ' '
                name = name + 'tot ' + str(tot)
            data.append([current_node['id'], name, current_node['parent'], 1])
    return data




def construct_data_tree_samples(model, y_predicted, n_leaves):
    list_nodes = [{'node':model.tree, 'id': 0, 'parent':None}]
    data = []
    i = 0
    labels_leaf = [i for i in range(n_leaves)]
    while len(list_nodes) != 0:
        current_node = list_nodes.pop(0)
        if current_node['node'].router is not None:
            data.append([current_node['id'], None, current_node['parent'], 'internal'])
            node_left, node_right = current_node['node'].left, current_node['node'].right
            i += 1
            list_nodes.append({'node':node_left, 'id': i, 'parent': current_node['id']})
            i += 1
            list_nodes.append({'node':node_right, 'id': i, 'parent': current_node['id']})
        elif current_node['node'].router is None and current_node['node'].decoder is None:
            # We are in an internal node with pruned leaves and will only add the non-pruned leaves
            data.append([current_node['id'], None, current_node['parent'], 'internal'])
            node_left, node_right = current_node['node'].left, current_node['node'].right
            child = node_left if node_left is not None else node_right
            i += 1
            list_nodes.append({'node': child, 'id': i, 'parent': current_node['id']})
        else:
            y_leaf = labels_leaf.pop(0)
            ind = np.where(y_predicted == y_leaf)[0]
            data.append([current_node['id'], ind, current_node['parent'], "leaf"])

    for node in reversed(data):
        if node[1] is None:
            node_id = node[0]
            child_ind = np.where([child_node[2]==node_id for child_node in data])[0]
            if len(child_ind)==1:
                node[1] = data[child_ind[0]][1]
            elif len(child_ind)==2:
                node[1] = np.sort(np.concatenate((data[child_ind[0]][1], data[child_ind[1]][1])))

    return data