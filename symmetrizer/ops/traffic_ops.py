import numpy as np
import torch
import torch.nn.functional as F

from .ops import GroupRepresentations


def stitch(tensor):
    """
    stitch 4 agents' states together into one big state
    """
    agent_obs = [tensor[:, i] for i in range(4)]
    bs, ag, ch, w, h = tensor.shape
    stitched = np.zeros((bs, ch, 1+w*2, 1+h*2))
    stitched[:, :, :w, :h] = agent_obs[0]
    stitched[:, :, w+1:, :h] = agent_obs[1]
    stitched[:, :, :w, h+1:] = agent_obs[2]
    stitched[:, :, w+1:, h+1:] = agent_obs[3]
    return torch.FloatTensor(stitched)


def get_traffic_rolls():
    perm = np.zeros((2, 2))

    permutes = [[1, 0],
                [0, 1],
                [1, 0]]

    perms = [perm.copy() for p in permutes]
    for i, per in enumerate(perms):
        p = permutes[i]
        for j in range(2):
            per[j][p[j]] = 1

    representations = [torch.FloatTensor(np.eye(2)),
                       torch.FloatTensor(perms[0]),
                       torch.FloatTensor(perms[1]),
                       torch.FloatTensor(perms[2])]
    return GroupRepresentations(representations, "TrafficRolls")


def get_traffic_agent_perms():
    perm = np.zeros((4, 4))
    permutes = [[2, 0, 3, 1],
                [3, 2, 1, 0],
                [1, 3, 0, 2]]
    perms = [perm.copy() for p in permutes]
    for i, per in enumerate(perms):
        p = permutes[i]
        for j in range(4):
            per[j][p[j]] = 1

    representations = [torch.FloatTensor(np.eye(4)),
                       torch.FloatTensor(perms[0]),
                       torch.FloatTensor(perms[1]),
                       torch.FloatTensor(perms[2])]
    return GroupRepresentations(representations, "TrafficAgentPerms")


def get_traffic_policy_perms():
    perm = np.zeros((8, 8))
    permutes = [[5, 4, 1, 0, 7, 6, 3, 2],
                [6, 7, 4, 5, 2, 3, 0, 1],
                [3, 2, 7, 6, 1, 0, 5, 4],]
    perms = [perm.copy() for p in permutes]
    for i, per in enumerate(perms):
        p = permutes[i]
        for j in range(8):
            per[j][p[j]] = 1
    representations = [torch.FloatTensor(np.eye(8)),
                       torch.FloatTensor(perms[0]),
                       torch.FloatTensor(perms[1]),
                       torch.FloatTensor(perms[2])]
    return GroupRepresentations(representations, "TrafficPolicyPerms")


def get_traffic_perms():
    perm = np.zeros((4, 4))
    permutes = [[1, 2, 3, 0],
                [2, 3, 0, 1],
                [3, 0, 1, 2]]
    perms = [perm.copy() for p in permutes]
    for i, per in enumerate(perms):
        p = permutes[i]
        for j in range(4):
            per[j][p[j]] = 1

    representations = [torch.FloatTensor(np.eye(4)),
                       torch.FloatTensor(perms[0]),
                       torch.FloatTensor(perms[1]),
                       torch.FloatTensor(perms[2])]
    return GroupRepresentations(representations, "TrafficPerms")


def get_wildlife_action_group_representations():
    """
    Representation of the group symmetry on the policy: a permutation of the
    actions
    """
    representations = [torch.FloatTensor(np.eye(5)),

                       torch.FloatTensor(np.array([[1, 0, 0, 0, 0],
                                                   [0, 0, 1, 0, 0],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 1],
                                                   [0, 1, 0, 0, 0]])),

                       torch.FloatTensor(np.array([[1, 0, 0, 0, 0],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 1],
                                                   [0, 1, 0, 0, 0],
                                                   [0, 0, 1, 0, 0]])),

                       torch.FloatTensor(np.array([[1, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 1],
                                                   [0, 1, 0, 0, 0],
                                                   [0, 0, 1, 0, 0],
                                                   [0, 0, 0, 1, 0]]))]
    return GroupRepresentations(representations, "VectorActionGroupRepr")


def get_4d_group_representations():
    """
    Representation of the group symmetry on the state: ...
    """
    representations = [torch.FloatTensor(np.eye(4)) for _ in range(4)]

    return GroupRepresentations(representations, "FeatureStateGroupRepr")


def get_feature_state_group_representations():
    representations = [torch.FloatTensor(np.eye(4)),

                       torch.FloatTensor(np.array([[0, 0, 0, 1],
                                                   [1, 0, 0, 0],
                                                   [0, 1, 0, 0],
                                                   [0, 0, 1, 0]])),

                       torch.FloatTensor(np.array([[0, 0, 1, 0],
                                                   [0, 0, 0, 1],
                                                   [1, 0, 0, 0],
                                                   [0, 1, 0, 0]])),

                       torch.FloatTensor(np.array([[0, 1, 0, 0],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, 1],
                                                   [1, 0, 0, 0]]))]

    return GroupRepresentations(representations, "FeatureStateGroupRepr")


def get_feature_action_group_representations():
    # Representation of the group symmetry on the policy: a permutation of the
    # actions
    representations = [torch.FloatTensor(np.eye(4)),

                       torch.FloatTensor(np.array([[0, 0, 0, 1],
                                                   [1, 0, 0, 0],
                                                   [0, 1, 0, 0],
                                                   [0, 0, 1, 0]])),

                       torch.FloatTensor(np.array([[0, 0, 1, 0],
                                                   [0, 0, 0, 1],
                                                   [1, 0, 0, 0],
                                                   [0, 1, 0, 0]])),

                       torch.FloatTensor(np.array([[0, 1, 0, 0],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, 1],
                                                   [1, 0, 0, 0]]))]
    return GroupRepresentations(representations, "FeatureActionGroupRepr")


def get_traffic_state_group_representations():
    rot_mat0 = torch.FloatTensor(np.eye(2))
    rot_mat270 = torch.FloatTensor(np.array([[0, -1], [1, 0]]))
    rot_mat180 = torch.FloatTensor(np.array([[-1, 0], [0, -1]]))
    rot_mat90 = torch.FloatTensor(np.array([[0, 1], [-1, 0]]))

    representations = [rot_mat0, rot_mat90, rot_mat180, rot_mat270]
    return GroupRepresentations(representations, "VectorStateGroupRepr")


def get_vector_invariants():
    """
    Function to enable easy construction of invariant layers (for value
    networks)
    """
    raise NotImplementedError()


def old_unnormalize(l):
    """
    """
    l += 1
    l /= 2
    l *= 5
    return l


def normalize(l, maximum=21., minimum=0., new_min=-1., new_max=1.):
    """
    [a, b]
    a + (x - min(x)) (b-a) / max(x) - min(x)
    """
    l = l - minimum
    l = l * (new_max - new_min)
    l = l / (maximum - minimum)
    return l


def locs2diffs(locs):
    """
    Transform list of locations into dictionary of pairwise difference vectors
    """
    n_agents = locs.shape[0]
    diffs = np.zeros((n_agents, n_agents, 2))
    for i in range(n_agents):
        for j in range(n_agents):
            diffs[i][j] = locs[i] - locs[j]
    return diffs


def get_locs(locs_grid, n_agents):
    """
    """
    if len(locs_grid.shape) > 2:
        locs = locs_grid[:, 0:n_agents, 0:2]
    else:
        locs = locs_grid[0:n_agents, 0:2]
    return locs


def get_diffs(locs):
    """
    diffs[:, i, j] -> x_i - x_j
    """
    diffs = locs.unsqueeze(2) - locs.unsqueeze(1)
    return diffs


def normalize_adjacency(A):
    # L1 normalize
    return F.normalize(A, dim=1, p=1)


def get_message(i, j, A, agent_features, func, msg_type, diff_vecs=None):
    if msg_type == "aggregate":
        msg_bool = A[:, i, j].unsqueeze(-1).unsqueeze(-1)
        f_j = agent_features[:, j].unsqueeze(1)
        return func(msg_bool * f_j)
    elif msg_type == "edge_agg":
        ch = agent_features.shape[2]
        msg_bool = A[:, i, j].unsqueeze(-1).unsqueeze(-1)
        f_j = agent_features[:, j]
        diff_ji = diff_vecs[:, j, i]
        f_hatji = torch.cat([diff_ji, f_j], dim=1).unsqueeze(1)
        return func(msg_bool * f_hatji)
    elif msg_type == "group_agg":
        ch = agent_features.shape[2]
        msg_bool = A[:, i, j].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        f_j = agent_features[:, j]
        diff_ji = diff_vecs[:, j, i].unsqueeze(1).repeat(1, ch, 1)
        f_hatji = torch.cat([diff_ji, f_j], dim=2).unsqueeze(1)
        return func(msg_bool * f_hatji)


def aggregate(msg_list, func, agg_type):
    if agg_type == "sum":
        return func(torch.sum(msg_list, dim=1))
    elif agg_type == "cat_sum":
        f_i, msgs = msg_list
        msg_sum = torch.sum(msgs, dim=1)
        total = torch.cat([f_i, msg_sum], dim=1)
        return func(total)
    elif agg_type == "group_cat_sum":
        f_i, msgs = msg_list
        msg_sum = torch.sum(msgs, dim=1)
        total = torch.cat([f_i, msg_sum], dim=2)
        return func(total)
