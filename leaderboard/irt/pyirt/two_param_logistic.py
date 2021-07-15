"""Copyright (c) Facebook, Inc. and its affiliates."""
# pylint: disable=unused-argument,unused-variable,not-callable,no-name-in-module,no-member,protected-access
from functools import partial

import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from pyro.infer import SVI, EmpiricalMarginal, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam
from rich.console import Console

console = Console()


class TwoParamLog:
    def __init__(self, priors, device, num_items, num_models, dims=None):
        self.priors = priors
        self.device = device
        self.num_items = num_items
        self.num_models = num_models

    def model_hierarchical(self, models, items, obs):
        """
        This model differs from the 1PL model by introducing a "discriminability"
        parameter.

        y ~ bernoulli_logit(gamma[kk] .* (alpha[jj] - (beta[kk] + mu_beta)));

        v,

        y[n] ~ bernoulli_logit(alpha[jj] - beta[kk] + mu_beta);
        """

        mu_b = pyro.sample(
            "mu_b",
            dist.Normal(
                torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device),
            ),
        )
        u_b = pyro.sample(
            "u_b",
            dist.Gamma(
                torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device),
            ),
        )

        mu_theta = pyro.sample(
            "mu_theta",
            dist.Normal(
                torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device),
            ),
        )
        u_theta = pyro.sample(
            "u_theta",
            dist.Gamma(
                torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device),
            ),
        )

        mu_gamma = pyro.sample(
            "mu_gamma",
            dist.Normal(
                torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device),
            ),
        )
        u_gamma = pyro.sample(
            "u_gamma",
            dist.Gamma(
                torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device),
            ),
        )

        with pyro.plate("thetas", self.num_models, device=self.device):
            ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))

        with pyro.plate("bs", self.num_items, device=self.device):
            diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))

        with pyro.plate("gammas", self.num_items, device=self.device):
            disc = pyro.sample("gamma", dist.Normal(mu_gamma, 1.0 / u_gamma))

        with pyro.plate("observe_data", obs.size(0)):
            pyro.sample(
                "obs",
                dist.Bernoulli(logits=disc[items] * (ability[models] - diff[items])),
                obs=obs,
            )

    def guide_hierarchical(self, models, items, obs):

        loc_mu_b_param = pyro.param("loc_mu_b", torch.tensor(0.0, device=self.device))
        scale_mu_b_param = pyro.param(
            "scale_mu_b", torch.tensor(1.0e2, device=self.device), constraint=constraints.positive,
        )
        loc_mu_gamma_param = pyro.param("loc_mu_gamma", torch.tensor(0.0, device=self.device))
        scale_mu_gamma_param = pyro.param(
            "scale_mu_gamma",
            torch.tensor(1.0e2, device=self.device),
            constraint=constraints.positive,
        )

        loc_mu_theta_param = pyro.param("loc_mu_theta", torch.tensor(0.0, device=self.device))
        scale_mu_theta_param = pyro.param(
            "scale_mu_theta",
            torch.tensor(1.0e2, device=self.device),
            constraint=constraints.positive,
        )

        alpha_b_param = pyro.param(
            "alpha_b", torch.tensor(1.0, device=self.device), constraint=constraints.positive,
        )
        beta_b_param = pyro.param(
            "beta_b", torch.tensor(1.0, device=self.device), constraint=constraints.positive,
        )

        alpha_gamma_param = pyro.param(
            "alpha_gamma", torch.tensor(1.0, device=self.device), constraint=constraints.positive,
        )
        beta_gamma_param = pyro.param(
            "beta_gamma", torch.tensor(1.0, device=self.device), constraint=constraints.positive,
        )

        alpha_theta_param = pyro.param(
            "alpha_theta", torch.tensor(1.0, device=self.device), constraint=constraints.positive,
        )
        beta_theta_param = pyro.param(
            "beta_theta", torch.tensor(1.0, device=self.device), constraint=constraints.positive,
        )
        m_theta_param = pyro.param("loc_ability", torch.zeros(self.num_models, device=self.device))
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones(self.num_models, device=self.device),
            constraint=constraints.positive,
        )
        m_b_param = pyro.param("loc_diff", torch.zeros(self.num_items, device=self.device))
        s_b_param = pyro.param(
            "scale_diff",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )
        m_gamma_param = pyro.param("loc_disc", torch.zeros(self.num_items, device=self.device))
        s_gamma_param = pyro.param(
            "scale_disc",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )

        # sample statements
        mu_b = pyro.sample("mu_b", dist.Normal(loc_mu_b_param, scale_mu_b_param))
        u_b = pyro.sample("u_b", dist.Gamma(alpha_b_param, beta_b_param))
        mu_gamma = pyro.sample("mu_gamma", dist.Normal(loc_mu_gamma_param, scale_mu_gamma_param))
        u_gamma = pyro.sample("u_gamma", dist.Gamma(alpha_gamma_param, beta_gamma_param))
        mu_theta = pyro.sample("mu_theta", dist.Normal(loc_mu_theta_param, scale_mu_theta_param))
        u_theta = pyro.sample("u_theta", dist.Gamma(alpha_theta_param, beta_theta_param))

        with pyro.plate("thetas", self.num_models, device=self.device):
            pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))

        with pyro.plate("bs", self.num_items, device=self.device):
            pyro.sample("b", dist.Normal(m_b_param, s_b_param))

        with pyro.plate("gammas", self.num_items, device=self.device):
            pyro.sample("gamma", dist.Normal(m_gamma_param, s_gamma_param))

    def fit(self, models, items, responses, num_epochs):
        optim = Adam({"lr": 0.1})
        svi = SVI(self.model_hierarchical, self.guide_hierarchical, optim, loss=Trace_ELBO(),)

        pyro.clear_param_store()
        for j in range(num_epochs):
            loss = svi.step(models, items, responses)
            if j % 100 == 0:
                console.log("[epoch %04d] loss: %.4f" % (j + 1, loss))

        console.log("[epoch %04d] loss: %.4f" % (j + 1, loss))
        values = ["loc_diff", "scale_diff", "loc_ability", "scale_ability"]

    def export(self):
        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "diff": pyro.param("loc_diff").data.tolist(),
            "disc": pyro.param("loc_disc").data.tolist(),
        }

    def fit_MCMC(self, models, items, responses, num_epochs):
        sites = ["theta", "b"]
        nuts_kernel = NUTS(self.model_vague, adapt_step_size=True)
        hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100).run(
            models, items, responses
        )
        theta_sum = self.summary(hmc_posterior, ["theta"]).items()
        b_sum = self.summary(hmc_posterior, ["b"]).items()
        console.log(theta_sum)
        console.log(b_sum)

    def summary(self, traces, sites):
        marginal = (
            EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
        )
        console.log(marginal)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            describe = partial(pd.Series.describe, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            site_stats[site_name] = marginal_site.apply(describe, axis=1)[
                ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
            ]
        return site_stats
