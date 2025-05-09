import numpy as np
import emcee

from bias_bench.optimizer.BaseOptimizer import BaseOptimizer


class EMCEESampler(BaseOptimizer):

    def __init__(self, likelihood, bias_model, n_walkers=50, n_steps=10000, n_discard=1000):
        super().__init__(likelihood, bias_model)

        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_discard = n_discard

    def log_prob(self, params, delta, data):
        likelihood_params = params[:self.like.n_params]
        bias_params = params[-self.model.n_params:]

        predict = self.model.predict(delta, bias_params)
        return - self.like.negLogLike(data, predict, likelihood_params)

    def sample(self, input_x, data, init_params):
        ndims = init_params.size
        # Initialize the walkers
        # TODO: Make random initialization respect bounds
        p0 = np.dot(np.ones((self.n_walkers, 1)), init_params.reshape(1, ndims)) + np.random.rand(self.n_walkers, ndims)

        # Set up the MCMC sampler
        sampler = emcee.EnsembleSampler(nwalkers=self.n_walkers, ndim=ndims, log_prob_fn=self.log_prob,
                                        args=[input_x, data])
        # Run the MCMC sampling
        sampler.run_mcmc(p0, self.n_steps, progress=True)
        samples = sampler.get_chain(discard=self.n_discard)
        log_probs = sampler.get_log_prob(discard=self.n_discard)

        return samples, log_probs

    def optimize(self, input_x, data, init_params):
        samples, log_probs = self.sample(input_x, data, init_params)
        res = np.mean(samples[:, :, :], axis=(0, 1))

        likelihood_params = res[:self.like.n_params]
        bias_params = res[-self.model.n_params:]
        return likelihood_params, bias_params, log_probs, samples
