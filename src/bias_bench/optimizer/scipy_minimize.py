from scipy.optimize import minimize

from bias_bench.optimizer.BaseOptimizer import BaseOptimizer


class BFGSScipy(BaseOptimizer):
    def __init__(self, likelihood, bias_model):
        super().__init__(likelihood, bias_model)

    def loss(self, delta, data, params):
        likelihood_params = params[:self.like.n_params]
        bias_params = params[-self.model.n_params:]

        predict = self.model.predict(delta, bias_params)
        return self.like.negLogLike(data, predict, likelihood_params)

    def print_opt_stats(self, delta, data, params):
        """
        Callback function to print the current parameters
        """
        err = self.loss(delta, data, params)
        print("Loss: {}".format(err))
        print("Parameters: {}".format(params))

    def optimize(self, delta, data, init_params):
        model_bounds = self.model.bounds
        res = minimize(lambda x: self.loss(delta, data, x), init_params,
                       method="L-BFGS-B", callback=lambda x: self.print_opt_stats(delta, data, x),
                       tol=1e-10,
                       bounds=model_bounds,
                       options={'disp': True, 'eps': 1e-5, 'finite_diff_rel_step': 1e-5})

        likelihood_params = res.x[:self.like.n_params]
        bias_params = res.x[-self.model.n_params:]
        return likelihood_params, bias_params
