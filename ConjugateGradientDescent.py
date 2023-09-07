import torch
from functools import reduce
from torch.optim.optimizer import Optimizer
from torch.autograd.functional import hvp
from pdb import set_trace

__all__ = ['ConjugateGradTR']

class ConjugateGradTR(Optimizer):
    """Implements Conjugate Gradient in a Trust-Region setting algorithm

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    For more information on the algorithm itself, please refer:
    https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf, Page 171 (CG-Steihaug)

    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 line_search_fn=None,
                 max_cg_iters = 10,
                 max_tr_iters = 10,
                 eta = 0.25,
                 deltaCap = 1):

        defaults = dict(
            lr=lr,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
            max_cg_iters=max_cg_iters,
            max_tr_iters=max_tr_iters,
            deltaCap=deltaCap,
            eta = eta,
            deltatr= deltaCap)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("ConjugateGradTR doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def _flatten_params(self):
        views = []
        for p in self._params:
            view = p.view(-1)
            views.append(view)
        
        return torch.cat(views, 0)


    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.add_(pdata)



    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad
    


    @torch.no_grad()
    def step(self, closure, model):
        """Performs a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_cg_iters = group['max_cg_iters']
        max_tr_iters = group['max_tr_iters']
        
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        deltaCap = group['deltaCap']
        deltatr = group['deltatr']
        eta = group['eta']

        # NOTE: ConjugateGradTR has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')
        
        '''
            We initialize our values for 
                z_0 = 0, 
                r_0 = \nabla f_k, 
                d_0 = -r_0 = -\nabla f_k 
        
        '''

        n_iter = 0
        epsilon = 1e-5
        # optimize for a max of max_iter iterations
        dj = flat_grad.neg()
        rj = flat_grad

        # Also, we are only doing it for a finite number of steps.
        tr_iterations = 0

        z = torch.zeros(self._flatten_params().shape)

        while tr_iterations< max_tr_iters:
            while n_iter < max_tr_iters:
                # Let's code the CG steihaug method here
                # Refer page 171 in https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
                n_iter += 1
                state['n_iter'] += 1
                set_trace()
                _,Bdj = hvp(model,inputs=self._flatten_params(), v=dj, create_graph=False)
                if dj.dot(Bdj) <=0:
                    # set the value of \tau such that 
                    # the value minimizes the loss along the direction of the last
                    # iterate
                    pass
                alpha = rj.dot(rj)/ dj.dot(Bdj)
                z = z + alpha*dj
                rj1 = rj + alpha* Bdj
                if torch.norm(rj1)< epsilon:
                    break
            
                betaj1 = rj1.dot(rj1)/rj.dot(rj)
                dj = -rj1 + betaj1*dj
                
            ############################################################
            # compute the trust-region step
            ############################################################
            # reset initial guess for step size

            _,Bz = hvp(model,inputs=self._flatten_params(), v=z, create_graph=False)
            g = self._gather_flat_grad()
            predDiff = g.dot(z) + 0.5*(z.dot(Bz))
            loss0 = closure()
            self._set_param(z)
            loss1 = closure()
            
            TDiff = loss1 - loss0
            rho = TDiff / predDiff
            
            if rho< 0.25:
                deltatr = 0.25*deltatr
            
            else:
                if rho > 0.75 and torch.norm(z) == delattr:
                    deltatr = max(2*deltatr, deltaCap)

            
            if rho < eta:
                self._set_param(-z)


            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_tr_iters:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if abs(loss - prev_loss) < tolerance_change:
                break

        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss
        state['deltatr'] = deltatr

        return orig_loss
