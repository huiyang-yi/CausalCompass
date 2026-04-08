import numpy as np
import torch
from scipy.special import expit as sigmoid  # same as notears
import igraph as ig
import random
import matplotlib.pyplot as plt
from .nts_notears import *
import copy
from lingam.utils import make_dot


def set_random_seed(seed):
    random.seed(seed)  # set python seed
    np.random.seed(seed)  # set numpy seed
    torch.manual_seed(seed)  # set pytorch seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs.


def write_log_file(file_name, text_str):
    f = open(file_name, "w")
    f.write(text_str)
    f.close()


def run_NTS_NOTEARS(normalized_X, data_column_names, number_of_lags, prior_knowledge, lambda1,
                    lambda2, w_threshold, h_tol, results_directory, device, network_dim, verbose=0, rho_max=1e+16,
                    validation_set=None):
    torch.set_default_dtype(torch.double)

    d = normalized_X.shape[1]

    variable_names_no_time = [s for s in data_column_names]

    model = NTS_NOTEARS(dims=[d] + network_dim, bias=True, number_of_lags=number_of_lags,
                        variable_names_no_time=variable_names_no_time, prior_knowledge=prior_knowledge)
    model = model.to(device)

    if verbose > 0:
        print('lambda1: ', lambda1)
        print('lambda2: ', lambda2)
        print('w_threshold: ', w_threshold)
        print('h_tol: ', h_tol)

    W_est_full = train_NTS_NOTEARS(model, normalized_X, device=device, lambda1=lambda1, lambda2=lambda2,
                                   w_threshold=w_threshold, h_tol=h_tol, verbose=verbose, rho_max=rho_max)

    # if cross-validation is performed, no need to draw estimated DAGs for each fold
    if validation_set is None:
        file_name = results_directory + 'estimated_DAG'

        variable_names = make_variable_names_with_time_steps(number_of_lags, data_column_names)
        if verbose > 0:
            print(variable_names)

        save_adjacency_matrix_in_csv(file_name, W_est_full, variable_names)

        draw_DAGs_using_LINGAM(file_name, W_est_full, variable_names)

        assert is_dag(W_est_full), 'The estimated graph has cycles.'

    # for validation
    if validation_set is not None:
        with torch.no_grad():
            W_simultaneous_est, W_NAR_est = model.fc1_to_adj()  # row -> column

            # eliminate edges smaller than `w_threshold`
            W_simultaneous_est[np.abs(W_simultaneous_est) < w_threshold] = 0
            W_NAR_est[np.abs(W_NAR_est) < w_threshold] = 0

            if number_of_lags != 1:
                raise Exception("current implementation for cross validation only supports K=1.")

            # set the corresponding weights on the 1D-CNN layer to 0s.
            # no need to set the biases to 0s.
            lag_kernel_index = 0
            simultaneous_kernel_index = 1
            for row in range(d):
                for column in range(d):

                    if W_NAR_est[row, column] == 0:
                        # CNN layer weights: (m * d, d, K+1)
                        model.conv1d_pos.weight[column * network_dim[0]:column * network_dim[0] + network_dim[0], row,
                        lag_kernel_index] = 0
                        model.conv1d_neg.weight[column * network_dim[0]:column * network_dim[0] + network_dim[0], row,
                        lag_kernel_index] = 0

                    if W_simultaneous_est[row, column] == 0:
                        # CNN layer weights: (m * d, d, K+1)
                        model.conv1d_pos.weight[column * network_dim[0]:column * network_dim[0] + network_dim[0], row,
                        simultaneous_kernel_index] = 0
                        model.conv1d_neg.weight[column * network_dim[0]:column * network_dim[0] + network_dim[0], row,
                        simultaneous_kernel_index] = 0

            validation_set_reshaped = reshape_for_model_forward(validation_set, model)
            X_val_torch = torch.from_numpy(validation_set_reshaped).to(device)

            X_val_hat = model(X_val_torch, device)

            validation_MSE = 2 * squared_loss(X_val_hat,
                                              torch.squeeze(X_val_torch.T)[
                                              model.simultaneous_idx:])  # compare the instantaneous time step

            # print("validation_MSE: ", validation_MSE.item())
            write_log_file(os.path.join(results_directory, "validation_MSE.txt"), str(validation_MSE.item()))

            model = None


def make_variable_names_with_time_steps(number_of_lags, data_column_names):
    """
    lagged W first, instantaneous W last, i.e.,

    ..., x1_{t-2}, x2_{t-2}, ..., x1_{t-1}, x2_{t-1}, ..., x1_{t}, x2_{t}, ...
    """
    variable_names = []
    for i in range(number_of_lags, 0, -1):
        variable_names_lagged = [s + '(t-{})'.format(i) for s in data_column_names]
        variable_names += variable_names_lagged

    variable_names_t = [s + '(t)' for s in data_column_names]
    variable_names += variable_names_t

    return variable_names


def save_adjacency_matrix_in_csv(file_name, adjacency_matrix, variable_names):
    """
    save the matrix in csv format with variable names
    """
    # create an empty matrix in object type (for string) with one extra row and column for variable names
    W_est_full_csv = np.array(np.zeros((len(variable_names) + 1, len(variable_names) + 1)), dtype=object)
    W_est_full_csv_binary = copy.deepcopy(W_est_full_csv)

    W_est_full_csv[0, 0] = W_est_full_csv_binary[0, 0] = 'row->column'
    W_est_full_csv[0, 1:] = W_est_full_csv_binary[0, 1:] = variable_names  # set column names
    W_est_full_csv[1:, 0] = W_est_full_csv_binary[1:, 0] = variable_names  # set row names

    # copy adjacency matrix
    # it is possible that the estimated adjacency matrix has less lags than the true lags
    # since the later lags is in the front of the full matrix, copy in the backward direction
    W_est_full_csv[-adjacency_matrix.shape[0]:, -adjacency_matrix.shape[1]:] = adjacency_matrix
    W_est_full_csv_binary[1:, 1:] = np.array(W_est_full_csv[1:, 1:] != 0, dtype=int)

    np.savetxt(file_name + '.csv', W_est_full_csv, delimiter=',', fmt='%s')

    np.savetxt(file_name + '_binary.csv', W_est_full_csv_binary, delimiter=',', fmt='%s')


def draw_DAGs_using_LINGAM(file_name, adjacency_matrix, variable_names):
    # direction of the adjacency matrix needs to be transposed.
    # in LINGAM, the adjacency matrix is defined as column variable -> row variable
    # in NOTEARS, the W is defined as row variable -> column variable

    # the default value here was 0.01. Instead of not drawing edges smaller than 0.01, we eliminate edges
    # smaller than `w_threshold` from the estimated graph so that we can set the value here to 0.
    lower_limit = 0.0

    # it is possible that the estimated adjacency matrix has less lags than the true lags
    # make up the size in this case
    if adjacency_matrix.shape[0] != len(variable_names) or adjacency_matrix.shape[1] != len(variable_names):
        W_est_full = np.array(np.zeros((len(variable_names), len(variable_names))))
        W_est_full[-adjacency_matrix.shape[0]:, -adjacency_matrix.shape[1]:] = adjacency_matrix
        adjacency_matrix = W_est_full

    dot = make_dot(np.transpose(adjacency_matrix), labels=variable_names, lower_limit=lower_limit)

    dot.format = 'png'
    dot.render(file_name)


def draw_adjacency_matrix_colormap(adjacency_matrix, total_d, title, result_folder):
    plt.matshow(adjacency_matrix, extent=[0, total_d, 0, total_d])

    major_ticks = np.linspace(0, total_d, total_d + 1)
    plt.xticks(major_ticks)
    plt.yticks(major_ticks)

    plt.grid(which="major", alpha=0.6)

    plt.tick_params(left=False,
                    top=False,
                    bottom=False,
                    labelleft=False,
                    labeltop=False,
                    labelbottom=False)

    plt.title(title, fontsize=20)

    # Saving the plot as an image
    plt.savefig(os.path.join(result_folder, title.replace(" ", "_") + "_Matrix_Colormap.png"))
    plt.close('all')  # close all plots, otherwise it would consume memory


def draw_adjacency_matrix_colormap_with_variables(adjacency_matrix, number_of_lags, d, title, result_folder,
                                                  variable_names, add_title=False):
    # https://stackoverflow.com/questions/38973868/adjusting-gridlines-and-ticks-in-matplotlib-imshow

    plt.figure()
    im = plt.imshow(adjacency_matrix[:, -d:], interpolation='none')

    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.gca.html
    ax = plt.gca()

    plt.tick_params(left=False,
                    top=False,
                    bottom=False,
                    labelleft=True,
                    labeltop=False,
                    labelbottom=True)

    # Labels for major ticks
    ax.set_xticklabels(variable_names[-d:], rotation=-90)
    ax.set_yticklabels(variable_names)

    total_d = d * (number_of_lags + 1)

    # Major ticks
    ax.set_xticks(np.arange(0, d, 1))
    ax.set_yticks(np.arange(0, total_d, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, d, 1), minor=True)
    ax.set_yticks(np.arange(-.5, total_d, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

    if add_title:
        plt.title(title, fontsize=20)

    # Saving the plot as an image
    plt.savefig(os.path.join(result_folder, title.replace(" ", "_") + "_Matrix_Colormap.png"), bbox_inches="tight")
    plt.close('all')  # close all plots, otherwise it would consume memory


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type, number_of_lags=0, average_degrees_per_lagged_node=[None, None]):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes (per time step)
        s0 (int): expected num of edges (for the instantaneous step)
        graph_type (str): ER (implemented), SF (implemented), BP
        number_of_lags: the total number of steps is (number_of_lags + 1)
        average_degrees_per_lagged_node: expected degrees of each node in [lag_2, lag_1], excludes the instantaneous
            step. It should be able to be used for both ER and SF.

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if len(average_degrees_per_lagged_node) != number_of_lags:
        raise Exception("what??? len(average_degree_per_lagged_node) {} != number_of_lags {}".format(
            len(average_degrees_per_lagged_node), number_of_lags))

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF' or graph_type == 'BA':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)

    if number_of_lags == 0:
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        return B_perm
    else:
        # Instantaneous edges are done.
        # Next, generate edges from lagged nodes to instantaneous nodes

        d_total = d * (number_of_lags + 1)
        B_time = np.zeros((d_total, d_total))

        # copy instantaneous edges
        B_time[-d:, -d:] = B_perm

        # generate edges from each lagged node to each instantaneous node based on the expected degree of each node at its lag
        for lag in range(number_of_lags, 0, -1):
            for from_node in range(0, d):
                # compute the index of the node based on its lag
                from_node_index = d * (number_of_lags - lag) + from_node

                # for each instantaneous node
                for to_node_index in range(-d, 0, 1):
                    # add an edge from `from_node` to `to_node` if the `random_number` is smaller than `threshold`.
                    random_number = np.random.uniform(low=0.0, high=1.0)
                    threshold = 1.0 / d * average_degrees_per_lagged_node[-lag]
                    if random_number <= threshold:
                        B_time[from_node_index, to_node_index] = 1

        assert ig.Graph.Adjacency(B_time.tolist()).is_dag()
        return B_time


def simulate_nonlinear_sem(B, n, sem_type, d, number_of_lags=0, noise_scale=None):
    def _simulate_single_equation_temporal(X, scale, w1=None, w2=None, w3=None):
        """X: [1, num of parents], x: [1]"""

        assert X.shape[0] == 1

        if sem_type == 'GLMPoissonDiscrete':
            z = np.random.randint(low=0, high=3, size=1)
        else:
            z = np.random.normal(scale=scale, size=1)

        pa_size = X.shape[1]
        if pa_size == 0:
            return z

        if sem_type == 'AdditiveNoiseModel':
            x = sigmoid(X @ w1) @ w2 + z

        elif sem_type == 'AdditiveIndexModel':
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z

        elif sem_type == 'GLMPoissonDiscrete':  # GLM with Poisson Distribution for discrete variables
            # https://jmlr.org/papers/volume20/18-819/18-819.pdf
            # Definition of Poisson DAG models:
            #   section 2.2, Y ~ Poisson(gj(XPa(j))) where for any arbitrary positive link function gj.
            # Identifiability:
            #   any Poisson DAG model is identifiable if all parents of node j contribute to its rate parameter.

            # using np.exp() as link function would create inf values.
            # According to the paper, Poisson DAGs are identifiable and Poisson DAGs can have any arbitrary positive link function.
            link_function = np.tanh(X @ w1) + 2
            x = np.random.poisson(link_function, size=1)

            if link_function <= 0:
                raise Exception("Link function must be positive but the link function is equal to: {}"
                                .format(link_function))

        else:
            raise ValueError('unknown sem type')
        return x

    scale_vec = noise_scale if noise_scale else np.ones(d)

    # i.i.d data points
    if number_of_lags == 0:
        raise Exception("time series data only.")

    # time series data
    else:
        assert B.shape[0] == B.shape[1]
        assert B.shape[0] > d
        assert B.shape[0] % d == 0

        G_all = ig.Graph.Adjacency(B.tolist())  # the graph containing both lagged and contemporaneous edges
        contemp_dag = ig.Graph.Adjacency(B[-d:, -d:].tolist())  # the graph containing only contemporaneous edges

        contemp_causal_order = contemp_dag.topological_sorting()

        assert len(contemp_causal_order) == d

        transient = int(.2 * n)  # the warm-up data points

        data = np.zeros((n + transient, d))

        # make initial data points using noise
        for t in range(number_of_lags):
            for j in range(d):
                parents = []
                data[t, j] = _simulate_single_equation_temporal(data[t, parents].reshape(1, -1), scale_vec[j])

        w_dict = dict()  # record the generated parameters of the SEM function for each variable
        # make data points using their corresponding parents
        for t in range(number_of_lags, n + transient):
            for j in contemp_causal_order:
                # contain both lagged and contemporaneous parents
                parents_all = G_all.neighbors(j + d * number_of_lags, mode=ig.IN)

                # contain only lagged parents
                parents_lagged = []
                for parent_index in parents_all:
                    max_lagged_parent_index = d * (number_of_lags + 1) - d - 1

                    if parent_index <= max_lagged_parent_index:
                        parents_lagged.append(parent_index)

                # contain only contemporaneous parents
                parents_contemp = contemp_dag.neighbors(j, mode=ig.IN)

                data_contemp = data[t, parents_contemp]

                data_lagged = []
                for parent_lagged in parents_lagged:
                    lag = number_of_lags - (parent_lagged // d)
                    corresponding_contemp = parent_lagged % d
                    data_current_lag = data[t - lag, corresponding_contemp]
                    data_lagged.append(data_current_lag)

                parents_data = np.array(data_lagged + data_contemp.tolist()).reshape((1, -1))

                # the SEM function for each variable j should be the same across all data points
                # therefore, only generates the function parameters once for each variable j
                # for each variable j in later data points, reuse the generated function parameters
                w1, w2, w3 = None, None, None

                # generate the function parameters
                if w_dict.get(j) is None:
                    pa_size = parents_data.shape[1]
                    if sem_type == 'AdditiveNoiseModel':
                        hidden = 100
                        w1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                        w1[np.random.rand(*w1.shape) < 0.5] *= -1
                        w2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                        w2[np.random.rand(hidden) < 0.5] *= -1
                    elif sem_type == 'AdditiveIndexModel' or sem_type == 'GLMPoissonDiscrete':
                        w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                        w1[np.random.rand(pa_size) < 0.5] *= -1
                        w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                        w2[np.random.rand(pa_size) < 0.5] *= -1
                        w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                        w3[np.random.rand(pa_size) < 0.5] *= -1

                    else:
                        raise Exception("sem_type '{}' is not supported.".format(sem_type))

                    w_sub_dict = dict()
                    w_sub_dict["w1"] = w1
                    w_sub_dict["w2"] = w2
                    w_sub_dict["w3"] = w3

                    assert (w1 is not None or w2 is not None or w3 is not None)

                    w_dict[j] = w_sub_dict

                # reuse the generated function parameters
                else:
                    w1 = w_dict[j]["w1"]
                    w2 = w_dict[j]["w2"]
                    w3 = w_dict[j]["w3"]

                    assert (w1 is not None or w2 is not None or w3 is not None)

                data[t, j] = _simulate_single_equation_temporal(parents_data, scale_vec[j], w1, w2, w3)

        data = data[transient:, :]
        return data


def count_accuracy(B_true, B_est, allow_cycles=False):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive (i.e. among the predicted edges,
            the percentage that are incorrect. The smaller the better.)
        precision: 1 - fdr, but base on the computation of fdr below, computing precision with fdr may produce
            problem when fdr=0. So compute precision without using fdr: (true positive) / prediction positive,
            (i.e. among the predicted edges, the percentage that are correct. The bigger the better.)
        tpr (recall): (true positive) / condition positive (i.e. among the true edges,
            the percentage that are predicted. The bigger the better.)
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')

        if not allow_cycles:
            # if not is_dag(B_est):
            #     raise ValueError('B_est should be a DAG')
            assert is_dag(B_est), 'B_est should be a DAG'

    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    # Refer to Linear Notears 2018, D.2. Metrics
    # fdr: False discovery rate, FDR = (R + FP) / P, the smaller the better
    # tpr: True positive rate, TPR = TP / T, the bigger the better
    # fpr: False positive rate, FPR = (R + FP) / F, the smaller the better
    # shd: Structural Hamming distance, SHD = E + M + R, the smaller the better, (total number of edge additions, deletions, and reversals needed to convert the estimated DAG into the true DAG)
    # nnz: number of predicted positives

    precision = float(len(true_pos)) / max(pred_size, 1)
    recall = tpr

    # https://en.wikipedia.org/wiki/F-score#Definition
    # The highest possible value of an F-score is 1.0, indicating perfect precision and recall,
    # and the lowest possible value is 0, if either the precision or the recall is zero.
    if precision == 0 or recall == 0:
        f1_score = 0.0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)

    return {'fdr': fdr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 'precision': precision, 'recall': recall,
            'f1_score': f1_score}
