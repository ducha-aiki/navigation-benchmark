import numpy as np

def calcSPL(coefs,successes):
    assert len(coefs) == len(successes)
    return np.mean(np.array(coefs) *np.array(successes))