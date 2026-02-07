"""

Reference:
    [1] https://github.com/jarrycyx/UNN

"""

def create_cutsplus_config(args):
    from types import SimpleNamespace

    opt = SimpleNamespace()
    opt.n_nodes = 'auto'
    opt.data_dim = 1
    opt.total_epoch = 100
    opt.batch_size = args.cutsplus_batch_size
    opt.input_step = args.cutsplus_input_step

    p = args.p
    if p == 10:
        opt.n_groups = 2
        opt.group_policy = "multiply_2_every_20"

    elif p == 15:
        opt.n_groups = 3
        opt.group_policy = "multiply_2_every_20"

    elif p == 20:
        opt.n_groups = 4
        opt.group_policy = "multiply_2_every_20"

    opt.fill_policy = "rate_0.1_after_20"
    opt.supervision_policy = "masked_before_100"
    opt.show_graph_every = 16

    opt.data_pred = SimpleNamespace()
    opt.data_pred.weight_decay = args.cutsplus_weight_decay

    # if args.data_model == 'VAR':
    #     opt.data_pred.weight_decay = 0.003
    #
    # elif args.data_model == 'Lorenz':
    #     opt.data_pred.weight_decay = 0

    opt.data_pred.model = 'multi_lstm'
    opt.data_pred.pred_step = 1
    opt.data_pred.mlp_hid = 32
    opt.data_pred.gru_layers = 1
    opt.data_pred.shared_weights_decoder = False
    opt.data_pred.concat_h = True
    opt.data_pred.lr_data_start = 1e-3
    opt.data_pred.lr_data_end = 1e-4
    opt.data_pred.prob = True

    opt.graph_discov = SimpleNamespace()
    opt.graph_discov.lambda_s_start = 1e-2
    opt.graph_discov.lambda_s_end = 1e-2
    opt.graph_discov.lr_graph_start = 1e-2
    opt.graph_discov.lr_graph_end = 1e-3
    opt.graph_discov.start_tau = 1
    opt.graph_discov.end_tau = 0.1
    opt.graph_discov.dynamic_sampling_milestones = [0]
    opt.graph_discov.dynamic_sampling_periods = [1]

    return opt

def load_missing_data_for_cutsplus(args, seed):
    import os
    import numpy as np

    if args.data_model == 'VAR':
        filename = f'missing_prob{args.missing_prob}_VAR_p{args.p}_T{args.T}_seed{seed}.npz'
    else:  # Lorenz
        F = getattr(args, 'F', 10)
        filename = f'missing_prob{args.missing_prob}_Lorenz_p{args.p}_T{args.T}_F{F}_seed{seed}.npz'

    filepath = os.path.join('datasets', 'missing', filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing data file not found: {filepath}")

    return np.load(filepath)