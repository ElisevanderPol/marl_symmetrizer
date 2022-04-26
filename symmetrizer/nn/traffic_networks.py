import torch
import torch.nn.functional as F
import numpy as np
from symmetrizer.ops.traffic_ops import get_diffs, normalize_adjacency, \
    get_message, aggregate, get_traffic_state_group_representations, \
    normalize, get_traffic_perms
from symmetrizer.groups import MatrixRepresentation
from symmetrizer.ops.ops import direct_sum_group
from symmetrizer.nn.modules import BasisLinear, BasisConv2d, GlobalMaxPool
from symmetrizer.ops import c2g
from symmetrizer.groups import P4, P4toInvariant, P4toTraffic
from symmetrizer.ops.wildlife_ops import get_grid_rolls


# >> Equivariant Networks <<

# --- Equivariant Container Module ---
class BasisDecentralizedModel(torch.nn.Module):
    """
    """
    def __init__(self, input_size, n_agents, hidden_sizes=[512],
                 channels=[16, 32], filters=[8, 5], strides=[1, 1],
                 paddings=[0, 0], basis="equivariant", out="equivariant"):
        super().__init__()
        self.encoder = BasisTrafficEncoder(input_size,
                                           hidden_sizes=hidden_sizes,
                                           channels=channels, filters=filters,
                                           strides=strides, paddings=paddings,
                                           gain_type="he", basis=basis,
                                           out=out)
        self.coordination = BasisTrafficCoordinator(int(channels[-1]/np.sqrt(4)),
                                                    hidden_sizes=hidden_sizes,
                                                    filters=filters,
                                                    strides=strides,
                                                    paddings=paddings,
                                                    gain_type="he",
                                                    basis=basis, out=out)

    def forward(self, diffs, states):
        """
        """
        states = self.encoder(states)
        policies, values = self.coordination(diffs, states)
        return policies, values


# --- Equivariant Encoder Network ---
class BasisTrafficEncoder(torch.nn.Module):
    """
    """
    def __init__(self, input_size, hidden_sizes=[512], channels=[16, 32],
                 filters=[8, 5], strides=[1, 1], paddings=[0, 0],
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        in_group = P4()

        layers = []
        for l, channel in enumerate(channels):
            c = int(channel/np.sqrt(4))
            f = (filters[l], filters[l])
            s = strides[l]
            p = paddings[l]
            if l == 0:
                first_layer = True
            else:
                first_layer = False

            conv = BasisConv2d(input_size, c, filter_size=f, group=in_group,
                               gain_type=gain_type, basis=basis,
                               first_layer=first_layer, padding=p, stride=s)
            layers.append(conv)
            input_size = c
        self.convs = torch.nn.ModuleList(layers)

        self.pool = GlobalMaxPool()

    def forward(self, states):
        """
        """
        bs, ag_ch, h, w = states.shape
        agents = 4
        ch = int(ag_ch//agents)
        states = states.reshape(bs*agents, ch, h, w)

        for i, c in enumerate(self.convs):
            states = F.relu(c(states))

        pool = c2g(self.pool(states), 4).squeeze(-1).squeeze(-1)
        _, ch, g = pool.shape
        pool = pool.reshape(bs, agents, ch, g)
        return pool


# --- Equivariant Coordinator Network ---
class BasisTrafficCoordinator(torch.nn.Module):
    """
    """
    def __init__(self, input_size, hidden_sizes=[512],
                 filters=[8, 5], strides=[1, 1], paddings=[0, 0],
                 gain_type='xavier', basis="equivariant", out="equivariant"):
        """
        """
        super().__init__()
        layers = []

        for l, channel in enumerate(hidden_sizes):
            c = int(channel/np.sqrt(4))

            g = BasisCatGraphLayer(input_size, c, gain_type=gain_type,
                                   basis=basis, bias_init=True)
            layers.append(g)
            input_size = c
        self.layers = torch.nn.ModuleList(layers)

        out_group = P4toTraffic()
        inv_group = P4toInvariant()
        self.fc4 = BasisLinear(input_size, 1, out_group, gain_type=gain_type,
                               basis=basis, bias_init=True)
        self.fc5 = BasisLinear(input_size, 1, inv_group, gain_type=gain_type,
                               basis=basis, bias_init=True)

    def forward(self, locs, states, thresh_vec=[1, 1]):
        """
        """
        diffs = get_diffs(locs)
        norms = torch.norm(diffs, dim=-1)

        threshold = torch.norm(torch.FloatTensor(thresh_vec).to(diffs.device))

        A = (norms < threshold).float()
        # L1 normalize
        A = normalize_adjacency(A)

        agents = locs.shape[1]

        for i, g in enumerate(self.layers):
            diffs, states = g(diffs, states, norms)
        policies = torch.cat([self.fc4(states[:, ag]) for ag in range(agents)],
                             dim=1)
        values = torch.cat([self.fc5(states[:, ag]) for ag in range(agents)],
                           dim=1)
        return policies, values


# --- Equivariant Node-Edge Layer ---
class BasisCatGraphLayer(torch.nn.Module):
    """
    """
    def __init__(self, input_size, output_size, basis="equivariant",
                 gain_type="xavier", bias_init=False, **kwargs):
        """
        """
        super().__init__()
        self.layer = SingleBasisDirectSumLayer(input_size, output_size,
                                               double_in=True,
                                               basis=basis,
                                               bias_init=bias_init,
                                               gain_type=gain_type, **kwargs)
        self.output_size = output_size

    def forward(self, diff_vecs, agent_features, A):
        """
        """
        next_features = []

        agents = A.shape[-1]
        next_features = []
        for i in range(agents):
            for j in range(agents):
                # Don't add local features to sum
                if j == i:
                    continue
                msgs = [get_message(i, j, A, agent_features, func=lambda x: x,
                                    msg_type="group_agg", diff_vecs=diff_vecs)
                        for j in range(agents)]
            f_i = agent_features[:, i]
            f_next_i = aggregate([f_i, torch.cat(msgs, 1)], func=self.layer,
                                 agg_type="group_cat_sum")
            next_features.append(f_next_i)
        out = torch.cat(next_features, dim=1)
        return diff_vecs, F.relu(out)


# --- Equivariant Direct Sum Layer ---
class SingleBasisDirectSumLayer(torch.nn.Module):
    """
    Single layer for cartpole symmetries
    """

    def __init__(self, input_size, output_size, double_in=False,
                 basis="equivariant",
                 gain_type="xavier", bias_init=False, bias=True, **kwargs):
        super().__init__()
        vector_representations = get_traffic_state_group_representations()
        feature_representations = get_grid_rolls()
        if double_in:
            in_group = direct_sum_group([feature_representations,
                                         vector_representations,
                                         feature_representations])
        else:
            in_group = direct_sum_group([vector_representations,
                                         feature_representations])
        out_group = get_traffic_perms()
        repr_in = MatrixRepresentation(in_group, out_group)

        self.fc1 = BasisLinear(input_size, output_size, group=repr_in,
                               basis=basis, gain_type=gain_type,
                               bias_init=bias_init, bias=bias)

    def forward(self, state):
        """
        """
        return self.fc1(state.unsqueeze(1))


# >> Standard Networks <<

# --- Standard Container Module ---
class StandardDecentralizedModel(torch.nn.Module):
    """
    """
    def __init__(self, input_size, n_agents, hidden_sizes=[512],
                 channels=[16, 32], filters=[8, 5], strides=[1, 1],
                 paddings=[0, 0]):
        super().__init__()
        global_pool = True
        self.encoder = StandardDecentralizedTrafficEncoder(input_size,
                                                           n_agents,
                                                           hidden_sizes=hidden_sizes,
                                                           channels=channels,
                                                           filters=filters,
                                                           strides=strides,
                                                           paddings=paddings)
        if global_pool:
            input_size = channels[-1]
        self.coordinator = StandardTrafficCoordinator(input_size, n_agents,
                                                      hidden_sizes=hidden_sizes)

    def forward(self, diffs, states):
        """
        """
        states = self.encoder(states)
        policies, values = self.coordinator(diffs, states)
        return policies, values


# --- Standard Encoder Network ---
class StandardDecentralizedTrafficEncoder(torch.nn.Module):
    """
    """
    def __init__(self, input_size, n_agents, hidden_sizes=[512],
                 channels=[16, 32], filters=[8, 5], strides=[1, 1],
                 paddings=[0, 0]):
        super().__init__()
        self.n_agents = n_agents

        input_size = 3

        layers = []

        for l, channel in enumerate(channels):
            c = channel
            f = (filters[l], filters[l])
            s = strides[l]
            p = paddings[l]

            conv = torch.nn.Conv2d(in_channels=input_size, out_channels=c,
                                   kernel_size=f, padding=p, stride=s)
            layers.append(conv)
            input_size = c
        self.convs = torch.nn.ModuleList(layers)

        self.pool = GlobalMaxPool()

    def forward(self, states):
        """
        """
        bs, ag_ch, h, w = states.shape
        agents = 4
        ch = int(ag_ch//agents)
        states = states.reshape(bs*agents, ch, h, w)
        for i, c in enumerate(self.convs):
            states = F.relu(c(states))
        pool = self.pool(states).squeeze(-1).squeeze(-1)
        _, ch = pool.shape
        pool = pool.reshape(bs, agents, ch)
        return pool


# --- Standard Coordinator Network ---
class StandardTrafficCoordinator(torch.nn.Module):
    """
    """
    def __init__(self, input_size, n_agents, hidden_sizes=[512],
                 filters=[8, 5], strides=[1, 1], paddings=[0, 0]):
        """
        """
        super().__init__()
        layers = []
        self.n_agents = n_agents
        for l, channel in enumerate(hidden_sizes):
            c = channel
            g = CatEdgeGraphLayer(input_size, c, bias_init=False)
            layers.append(g)
            input_size = c
        self.layers = torch.nn.ModuleList(layers)

        self.fc4 = torch.nn.Linear(input_size, 2)
        self.fc5 = torch.nn.Linear(input_size, 1)

    def forward(self, locs, states, thresh_vec=[1, 1]):
        """
        """
        diffs = get_diffs(locs)
        norms = torch.norm(diffs, dim=-1)
        threshold = torch.norm(normalize(
            torch.FloatTensor(thresh_vec).to(diffs.device)))

        A = (norms < threshold).float()
        # L1 normalize
        A = normalize_adjacency(A)

        for i, g in enumerate(self.layers):
            diffs, states = g(diffs, states, A)

        policies = torch.cat([self.fc4(states[:, ag]).unsqueeze(1)
                              for ag in range(self.n_agents)], dim=1)
        values = torch.cat([self.fc5(states[:, ag]).unsqueeze(1)
                            for ag in range(self.n_agents)], dim=1)
        return policies, values


# --- Standard Node-Edge Layer ---
class CatEdgeGraphLayer(torch.nn.Module):
    """
    """
    def __init__(self, input_size, output_size, bias_init=False):
        """
        """
        super().__init__()
        self.layer = torch.nn.Linear((2*input_size)+2, output_size, bias=True)
        self.output_size = output_size

    def forward(self, diff_vecs, agent_features, A):
        """
        """
        next_feat = []
        agents = A.shape[1]
        for i in range(agents):
            msgs = [get_message(i, j, A, agent_features, func=lambda x:x,
                                msg_type="edge_agg", diff_vecs=diff_vecs)
                    for j in range(agents) if i != j]
            f_i = agent_features[:, i]
            f_next_i = aggregate([f_i, torch.cat(msgs, 1)], func=self.layer,
                                 agg_type="cat_sum")
            next_feat.append(f_next_i.unsqueeze(1))
        out = torch.cat(next_feat, dim=1)
        return diff_vecs, F.relu(out)
