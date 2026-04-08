import torch
import scipy.optimize as sopt


class LBFGSBScipy(torch.optim.Optimizer):
    """Wrap L-BFGS-B algorithm, using scipy routines.
    
    Courtesy: Arthur Mensch's gist
    https://gist.github.com/arthurmensch/c55ac413868550f89225a0b9212aa4cd
    """

    def __init__(self, params):
        defaults = dict()
        super(LBFGSBScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGSBScipy doesn't support per-parameter options"
                             " (parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel = sum([p.numel() for p in self._params])

        self.conv1d_pos_instantaneous_bounds = None
        self.conv1d_neg_instantaneous_bounds = None
        self.conv1d_pos_lag_bounds_lists = None
        self.conv1d_neg_lag_bounds_lists = None

        self.model_dims = None
        self.kernal_size = None

    def assign_bounds(self, model):
        self.conv1d_pos_instantaneous_bounds = model.conv1d_pos.instantaneous_bounds
        self.conv1d_neg_instantaneous_bounds = model.conv1d_neg.instantaneous_bounds
        self.conv1d_pos_lag_bounds_lists = model.conv1d_pos.lag_bounds_lists
        self.conv1d_neg_lag_bounds_lists = model.conv1d_neg.lag_bounds_lists

        self.model_dims = model.dims
        self.kernal_size = model.kernal_size

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_bounds(self):
        pos_is_set = False
        bounds = []
        for p in self._params:

            # set the bounds for the pos weights, lagged first, then instantaneous.
            if p.size() == (
                    self.model_dims[0] * self.model_dims[1], self.model_dims[0], self.kernal_size) and not pos_is_set:
                b = []
                for i in range(len(self.conv1d_pos_lag_bounds_lists)):
                    b = b + self.conv1d_pos_lag_bounds_lists[i] + [self.conv1d_pos_instantaneous_bounds[i]]

                pos_is_set = True

            # set the bounds for the neg weights, lagged first, then instantaneous.
            elif p.size() == (
                    self.model_dims[0] * self.model_dims[1], self.model_dims[0], self.kernal_size) and pos_is_set:
                b = []
                for i in range(len(self.conv1d_neg_lag_bounds_lists)):
                    b = b + self.conv1d_neg_lag_bounds_lists[i] + [self.conv1d_neg_instantaneous_bounds[i]]

            # set the bounds for biases or weights not in the input layer
            else:
                b = [(None, None)] * p.numel()

            bounds += b
        return bounds

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset:offset + numel].view_as(p.data)
            offset += numel
        assert offset == self._numel

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params)
            flat_params = flat_params.to(torch.get_default_dtype())
            self._distribute_flat_params(flat_params)
            loss = closure()
            loss = loss.item()
            flat_grad = self._gather_flat_grad().cpu().detach().numpy()
            return loss, flat_grad.astype('float64')

        initial_params = self._gather_flat_params()
        initial_params = initial_params.cpu().detach().numpy()

        bounds = self._gather_flat_bounds()

        # Magic
        sol = sopt.minimize(wrapped_closure,
                            initial_params,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=bounds)

        final_params = torch.from_numpy(sol.x)
        final_params = final_params.to(torch.get_default_dtype())
        self._distribute_flat_params(final_params)
