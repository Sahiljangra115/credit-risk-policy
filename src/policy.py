"""Cost-based approval threshold. Per approved loan with exposure E and default probability p:
expected profit = (1 - p) * MARGIN * E - p * LGD * E. Rejected loans earn 0."""
import numpy as np
from . import config as C


def loan_profit(pd_hat, exposure, margin=C.MARGIN, lgd=C.LGD):
    return (1 - pd_hat) * margin * exposure - pd_hat * lgd * exposure


def realized_profit(y, exposure, approve, margin=C.MARGIN, lgd=C.LGD):
    """Profit using true outcomes y (1 = default) over approved loans only."""
    return float(np.sum(np.where(approve, np.where(y == 1, -lgd, margin) * exposure, 0.0)))


def breakeven_pd(margin=C.MARGIN, lgd=C.LGD):
    """Approving is positive-EV iff p < margin / (margin + lgd)."""
    return margin / (margin + lgd)


def best_threshold(pd_hat, exposure, margin=C.MARGIN, lgd=C.LGD):
    """Cut-off on PD maximising expected profit on a (calibration) sample. Approve if pd <= cut."""
    order = np.argsort(pd_hat)
    cum = np.cumsum(loan_profit(pd_hat[order], exposure[order], margin, lgd))
    i = int(np.argmax(cum))
    if cum[i] <= 0:
        return -np.inf
    return float(pd_hat[order][i])
