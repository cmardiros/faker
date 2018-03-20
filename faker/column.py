import logging
import numpy as np

from faker import utils, pool


class FakeColumnGenerator(object):
    """
    Generate an column of fake data by resampling from a pool of data.

    Options:
    - control the observed frequency of values
    - control the ratio of nulls
    - control the ratio of zeros
    """

    def __init__(self,
                 rows,
                 nan_ratio=None,
                 zero_ratio=None):

        self.rows = rows
        self.nan_ratio = nan_ratio or 0
        self.zero_ratio = zero_ratio or 0

        self.nan_rows = int(self.rows * (self.nan_ratio * 1.0))
        self.zero_rows = int(self.rows * (self.zero_ratio * 1.0))
        self.filled_rows = self.rows - (self.nan_rows + self.zero_rows)

    def generate(self,
                 pool=None,
                 pool_size=None,
                 pool_kind=None,
                 pool_unique=False,
                 pool_kws=None,
                 pool_shuffle=False,
                 pool_sort=False,
                 pool_sort_order=None,
                 freq_probs=None,
                 freq_probs_dist_kws=None,
                 resample=True,
                 conditional=False,
                 unique=False,
                 shuffle=False,
                 sort=False,
                 sort_order=None,
                 seed=12345):
        """
        Generate a column of values by:

        1. resampling from a pool of values (given or generated)
        that occur with certain probability (given or generated).

        2. setting directly to be equal to the pool of values
        (given or generated) -- this is useful when the resulting data is
        conditional on the pool and we need to preserve the order
        """

        if conditional:
            # if this column is conditional on some other data
            # we may not wish to resample or sort/shuffle or distinctify
            # so override any args that try to do that
            pool_unique = False
            pool_shuffle = False
            pool_sort = False
            resample = False
            unique = False
            shuffle = False
            sort = False

        prng = np.random.RandomState(seed=seed)

        # generate a pool of values to resample from if one not given
        pool = (pool if pool is not None
                else self.generate_pool(size=pool_size,
                                        kind=pool_kind,
                                        unique=pool_unique,
                                        seed=seed,
                                        **(pool_kws or dict())))
        n_pool = pool.shape[0]

        pool = self.prepare_pool(pool=pool,
                                 pool_sort=pool_sort,
                                 pool_sort_order=pool_sort_order,
                                 pool_shuffle=pool_shuffle,
                                 seed=seed)

        # print("pool: {}".format(pool))

        # initiate our column with empty values and fill with nans
        x = np.empty(shape=self.rows, dtype=object)
        x.fill(np.nan)

        # fill with zeroes
        if self.zero_rows > 0:
            x[0:self.zero_rows] = np.zeros(shape=(self.zero_rows, ),
                                           dtype=object)

        if resample:
            # get frequency probabilities for resampling
            if freq_probs:
                pass
            elif freq_probs_dist_kws:
                freq_probs = self.generate_probs(size=n_pool,
                                                 dist_kws=freq_probs_dist_kws,
                                                 seed=seed)
            else:
                # uniform probabilities
                freq_probs = np.full(shape=(n_pool, ), fill_value=1.0 / n_pool)

            # resample non-null values up to self.filled_rows
            xfilled = prng.choice(pool,
                                  size=self.filled_rows,
                                  replace=False if unique else True,
                                  p=freq_probs)
        else:
            if n_pool != self.filled_rows:
                raise Exception("n_pool != self.filled_rows, {} != {}".format(
                    n_pool,
                    self.filled_rows))

            xfilled = pool

        if sort:
            xfilled = np.sort(xfilled)

            if sort_order == 'desc':
                xfilled = np.flip(xfilled, axis=0)

        # fill our array
        x[self.zero_rows:(self.zero_rows + self.filled_rows)] = xfilled

        if shuffle:
            prng.shuffle(x)

        return x

    def generate_pool(self,
                      size,
                      kind,
                      unique,
                      seed,
                      **kws):
        """
        Generate pool of values to resample from.
        """

        size = size or 1

        # calc pool size
        if size > 1:
            pass
        elif size == 1:
            size = self.filled_rows
        elif (size < 1 and size > 0):
            size = int(float(self.filled_rows) * size)

        logging.debug("generate resample pool of: {}".format(size))

        return pool.FakePoolGenerator(size=size,
                                      unique=unique).generate(seed=seed,
                                                              kind=kind,
                                                              **kws)

    def prepare_pool(self,
                     pool,
                     pool_sort,
                     pool_sort_order,
                     pool_shuffle,
                     seed):

        prng = np.random.RandomState(seed=seed)

        if pool_sort:
            pool = np.sort(pool)

            if pool_sort_order == 'desc':
                pool = np.flip(pool, axis=0)

        if pool_shuffle:
            prng.shuffle(pool)

        return pool

    def generate_probs(self,
                       size,
                       dist_kws,
                       seed):
        """
        Generate probabilities for resampling from the pool.
        """

        return utils.generate_probs(size=size,
                                    dist_kws=dist_kws,
                                    seed=seed)
