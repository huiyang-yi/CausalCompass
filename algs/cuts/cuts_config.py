"""

Reference:
    [1] https://github.com/jarrycyx/UNN

"""

def create_cuts_config(args):

    from types import SimpleNamespace

    opt = SimpleNamespace()

    opt.batch_size = args.cuts_batch_size
    opt.input_step = args.cuts_input_step

    opt.n_nodes = args.p
    opt.data_dim = 1
    opt.total_epoch = 100
    opt.supervision_policy = 'masked_before_50'
    opt.fill_policy = 'rate_0.1_after_10'
    opt.show_graph_every = 20

    opt.data_pred = SimpleNamespace()
    opt.data_pred.weight_decay = args.cuts_weight_decay
    # if args.data_model == 'VAR':
    #     opt.data_pred.weight_decay = 0.001
    #
    # elif args.data_model == 'Lorenz':
    #     opt.data_pred.weight_decay = 0
    opt.data_pred.model = 'multi_mlp'
    opt.data_pred.pred_step = 1
    opt.data_pred.mlp_hid = 128
    opt.data_pred.mlp_layers = 3
    opt.data_pred.lr_data_start = 1e-4
    opt.data_pred.lr_data_end = 1e-5
    opt.data_pred.prob = True

    opt.graph_discov = SimpleNamespace()
    opt.graph_discov.lambda_s_start = 0.1
    opt.graph_discov.lambda_s_end = 0.1
    opt.graph_discov.lr_graph_start = 1e-2
    opt.graph_discov.lr_graph_end = 1e-3
    opt.graph_discov.start_tau = 1
    opt.graph_discov.end_tau = 0.1
    opt.graph_discov.dynamic_sampling_milestones = [0]
    opt.graph_discov.dynamic_sampling_periods = [1]

    opt.causal_thres = 'value_0.5'

    return opt


def load_missing_data_for_cuts(args, seed):

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

    data_dict = np.load(filepath)

    return {
        'data_interp':data_dict['data_interp'],
        'original_data':data_dict['original_data'],
        'mask':data_dict['mask'],
        'gc':data_dict['gc']
    }
