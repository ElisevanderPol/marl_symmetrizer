import numpy as np
import torch
import torch.nn.functional as F

from .ops import GroupRepresentations


def get_grid_rolls():
    """Get permutations for the grid."""
    perm = np.zeros((4, 4))
    permutes = [[3, 0, 1, 2],
                [2, 3, 0, 1],
                [1, 2, 3, 0]]
    perms = [perm.copy() for p in permutes]
    for i, per in enumerate(perms):
        p = permutes[i]
        for j in range(4):
            per[j][p[j]] = 1

    representations = [torch.FloatTensor(np.eye(4)),
                       torch.FloatTensor(perms[0]),
                       torch.FloatTensor(perms[1]),
                       torch.FloatTensor(perms[2])]
    return GroupRepresentations(representations, "GridRolls")


def get_grid_actions():
    perm = np.zeros((5, 5))
    permutes = [[0, 4, 1, 2, 3],
                [0, 3, 4, 1, 2],
                [0, 2, 3, 4, 1]]
    perms = [perm.copy() for p in permutes]
    for i, per in enumerate(perms):
        p = permutes[i]
        for j in range(5):
            per[j][p[j]] = 1

    representations = [torch.FloatTensor(np.eye(5)),
                       torch.FloatTensor(perms[0]),
                       torch.FloatTensor(perms[1]),
                       torch.FloatTensor(perms[2])]
    return GroupRepresentations(representations, "GridActions")


def get_wildlife_rolls():
    """Get rolling permutations for wildlifes environment."""
    perm = np.zeros((5, 5))
    permutes = [[0, 2, 3, 4, 1],
                [0, 3, 4, 1, 2],
                [0, 4, 1, 2, 3]]
    perms = [perm.copy() for p in permutes]
    for i, per in enumerate(perms):
        p = permutes[i]
        for j in range(5):
            per[j][p[j]] = 1

    representations = [torch.FloatTensor(np.eye(5)),
                       torch.FloatTensor(perms[0]),
                       torch.FloatTensor(perms[1]),
                       torch.FloatTensor(perms[2])]
    return GroupRepresentations(representations, "WildlifeRolls")

def get_wildlife_perms():
    """Get permutations for wildlifes environment."""
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
    return GroupRepresentations(representations, "WildlifeRolls")



def normalize(l, maximum=21., minimum=0., new_min=-1., new_max=1.):
    """
    [a, b]
    a + (x - min(x)) (b-a) / max(x) - min(x)
    """
    l = l - minimum
    l = l * (new_max - new_min)
    l = l / (maximum - minimum)
    return l


def get_vector_state_group_representations():
    """
    Representation of the group symmetry on the state: ...
    """
    rot_mat0 = torch.FloatTensor(np.eye(2))
    rot_mat90 = torch.FloatTensor(np.array([[0, -1], [1, 0]]))
    rot_mat180 = torch.FloatTensor(np.array([[-1, 0], [0, -1]]))
    rot_mat270 = torch.FloatTensor(np.array([[0, 1], [-1, 0]]))

    representations = [rot_mat0, rot_mat90, rot_mat180, rot_mat270]
    return GroupRepresentations(representations, "VectorStateGroupRepr")

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
        locs = normalize(locs_grid[:, 0:n_agents, 0:2])
    else:
        locs = normalize(locs_grid[0:n_agents, 0:2])

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
    # Aggregate edge wise
    if msg_type == "edge_agg":
        ch = agent_features.shape[2]
        msg_bool = A[:, i, j].unsqueeze(-1).unsqueeze(-1)
        f_j = agent_features[:, j]
        diff_ji = diff_vecs[:, j, i]
        f_hatji = torch.cat([diff_ji, f_j], dim=1).unsqueeze(1)
        return func(msg_bool * f_hatji)
    elif msg_type == "group_agg":
    # Aggregate edge wise, group reps
        ch = agent_features.shape[2]
        msg_bool = A[:, i, j].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        f_j = agent_features[:, j]
        diff_ji = diff_vecs[:, j, i].unsqueeze(1).repeat(1, ch, 1)
        f_hatji = torch.cat([diff_ji, f_j], dim=2).unsqueeze(1)
        return func(msg_bool * f_hatji)


def aggregate(msg_list, func, agg_type):
    # Concatenate & aggregate for standard reps
    if agg_type == "cat_sum":
        f_i, msgs = msg_list
        msg_sum = torch.sum(msgs, dim=1)
        total = torch.cat([f_i, msg_sum], dim=1)
        return func(total)
    # Concatenate & aggregate for group reps
    elif agg_type == "group_cat_sum":
        f_i, msgs = msg_list
        msg_sum = torch.sum(msgs, dim=1)
        total = torch.cat([f_i, msg_sum], dim=2)
        return func(total)
