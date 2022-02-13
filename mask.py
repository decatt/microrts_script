import torch
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], use_gpu=False):
        self.masks = masks
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(self.device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(self.device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    # H = sum(p(x)log(p(x)))
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)

