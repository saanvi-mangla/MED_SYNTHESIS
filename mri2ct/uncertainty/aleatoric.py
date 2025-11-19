import torch
import torch.nn as nn
class AleatoricUncertainty:
    def __init__(self, model_with_variance):
        self.model = model_with_variance
    @torch.no_grad()
    def predict_with_uncertainty(self, x, context=None):
        self.model.eval()
        if context is not None:
            mean, log_var = self.model(x, context)
        else:
            mean, log_var = self.model(x)
        aleatoric_var = torch.exp(log_var)
        return mean, aleatoric_var
class CombinedUncertainty:
    def __init__(self, mc_dropout_model, num_mc_samples=20):
        self.mc_dropout = MCDropout(mc_dropout_model, num_mc_samples)
    @torch.no_grad()
    def predict_with_uncertainty(self, x, context=None, progress=True):
        mean, total_var, samples = self.mc_dropout.predict_with_uncertainty(
            x, context, progress
        )
        epistemic_var = total_var
        aleatoric_var = torch.zeros_like(total_var)
        return mean, epistemic_var, aleatoric_var, total_var
from uncertainty.mc_dropout import MCDropout
