import numpy as np
import emcee

from bias_bench.optimizer.BaseOptimizer import BaseOptimizer


class EMCEESampler(BaseOptimizer):

    def __init__(self, likelihood, bias_model):
        super().__init__(likelihood, bias_model)

        # TODO: make this adjustable
        self.nwalkers = 50
        self.nsteps = 2500
        self.ndiscard = 500

    def loss(self, params, delta, data):
        likelihood_params = params[:self.like.n_params]
        bias_params = params[-self.model.n_params:]

        predict = self.model.predict(delta, bias_params)
        return self.like.negLogLike(data, predict, likelihood_params)

    def sample(self, input_x, data, init_params):
        ndims = init_params.size
        # Initialize the walkers
        # Input parameters = POSTERIOR, HALOS, DELTA, ndim, nwalkers = 25, nsteps = 2500, discard = 200
        # TODO: Make random initialization respect bounds
        p0 = np.dot(np.ones((self.nwalkers, 1)), init_params.reshape(1, ndims)) + np.random.rand(self.nwalkers, ndims)

        # Set up the MCMC sampler
        sampler = emcee.EnsembleSampler(self.nwalkers,  # number of independent chains
                                        ndims,  # number of MCMC parameters
                                        self.loss,  # the posterior distribution to sample through
                                        # arguments that go into POSTERIOR
                                        args=(input_x,  # Halo data
                                              data))  # Delta_m data
        # Run the MCMC sampling
        sampler.run_mcmc(p0, self.nsteps, progress=True)
        # Get the chain samples; discard = discards the first number of samples
        samples = sampler.get_chain(discard=self.ndiscard)
        # dimensions: (samples, nwalker_index, MCMC parameter)
        return np.mean(samples[:, :, :], axis=(0, 1))  # we take the mean over the samples for each chain,
        # and return that as the estimate for the parameters.
        # How else could we do an estimate for the inferred parameters?

    def optimize(self, input_x, data, init_params):
        res = self.sample(input_x, data, init_params)

        likelihood_params = res[:self.like.n_params]
        bias_params = res[-self.model.n_params:]
        return likelihood_params, bias_params
