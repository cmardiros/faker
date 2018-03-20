import numpy as np


class ExactBernoulliSequence(object):
    """
    Not a distribution but a scipy dist-like object to generate a 1-d numpy
    array of `sample_size` with `successes` ones and the rest zeros.

    Useful for resampling simulations for proportions where we need to
    control the number of successes precisely

    NB: Unlike scipy.stats.bernoulli, this returns sequence with exactly
    the given number of successes.
    """

    def __init__(self,
                 successes=None,
                 p=None):

        self.successes = successes
        self.p = p

    def rvs(self,
            size,
            shuffle=True,
            random_state=None):

        if self.successes is None:
            successes = int(np.ceil(size * self.p))
        else:
            successes = self.successes

        zeros = np.zeros((size - successes,))
        ones = np.ones((successes,))

        sample = np.concatenate((zeros, ones))

        if shuffle:
            prng = np.random.RandomState(random_state)
            prng.shuffle(sample)

        return sample
