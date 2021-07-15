"""Copyright (c) Facebook, Inc. and its affiliates."""
import pyro
from py_irt.models.one_param_logistic import OneParamLog


class OneParamLogWithExport(OneParamLog):
    # pylint: disable=unused-argument
    def __init__(self, priors, device, num_items, num_models, dims=None):
        super().__init__(priors, device, num_items, num_models)

    def export(self):
        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "diff": pyro.param("loc_diff").data.tolist(),
        }
