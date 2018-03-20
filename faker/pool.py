import logging
import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats

from faker import utils, one


class FakePoolGenerator(object):
    """
    Generate an array of fake data.

    This is a building block for creating fake columns of data as it acts
    as a pool of values to resample from.
    """

    def __init__(self,
                 size=None,
                 unique=False):

        self.unique = unique

        # size of array to generate
        # this can be passed as None if generation method doesn't require it
        self.size = size

    def generate(self, kind, seed, **kws):

        mapping = {
            'from_function': self.from_function,
            'ints': self.ints,
            'counts': self.counts,
            'floats': self.floats,
            'strings': self.strings,
            'ints_sequence': self.ints_sequence,
            'dates': self.dates,
            'tstamps': self.tstamps,
        }

        # these methods already create unique sequences
        natively_unique = ['ints_sequence', 'dates']

        if self.unique and kind not in natively_unique:
            # increase the chance of generating unique values
            iters = self.size * 3
        else:
            iters = self.size

        func = mapping[kind]

        data = func(size=iters, seed=seed, **kws)

        if kind in natively_unique:
            return np.array(data)
        else:
            if self.unique:
                # resample without replacement to get uniques
                prng = np.random.RandomState(seed=seed)
                return prng.choice(data, size=self.size, replace=False)
            else:
                return np.array(data)

    def from_function(self, size, seed, **kws):

        return utils.from_function(iters=size, seed=seed, **kws)

    def strings(self, size, seed, **kws):

        kws = kws or dict()
        kws = dict({'length': 10, 'letters': True, 'digits': True}, **kws)

        return utils.from_function(
            iters=size,
            func=one.FakeStringValueGenerator(**kws).generate,
            func_seed_arg='seed',
            seed=seed)

    def ints(self, size, seed, **kws):

        kws = kws or dict()
        kws = dict({'low': 1, 'high': 10000}, **kws)

        return utils.from_function(iters=size,
                                   func=stats.randint.rvs,
                                   func_kws=kws,
                                   func_seed_arg='random_state',
                                   seed=seed)

    def counts(self, size, seed, **kws):
        """
        Counts are some of the most ubiquitous type of digital data so
        let's have a convenience function that generates realistic counts.
        """
        kws = kws.get('gamma_kws') or dict()
        kws = dict({'a': 2, 'c': 0.4, 'loc': 0, 'scale': 4}, **kws)
        # some good combinations to give skewed counts with 1/2 having most
        # frequency and a long tail are
        # {'a': 2, 'c': 0.4, 'loc': 0, 'scale': 4}
        # {'a': 2, 'c': 0.5, 'loc': 0, 'scale': 4}
        # {'a': 1.5, 'c': 0.4, 'loc': 0, 'scale': 4}
        # {'a': 1.5, 'c': 0.5, 'loc': 0, 'scale': 4}
        # {'a': 1, 'c': 0.4, 'loc': 0, 'scale': 4}
        # {'a': 1, 'c': 0.5, 'loc': 0, 'scale': 4}

        data = utils.from_function(iters=size,
                                   func=stats.gengamma.rvs,
                                   func_kws=kws,
                                   func_seed_arg='random_state',
                                   seed=seed)

        data = data + 1
        data = np.around(data, decimals=0)

        return data

    def ints_sequence(self, size, seed, **kws):

        kws = kws or dict()

        if kws.get('start'):
            kws['stop'] = kws['start'] + size

        elif kws.get('stop'):
            kws['start'] = kws['stop'] - size

        kws['step'] = 1

        return np.arange(**kws)

    def floats(self, size, seed, **kws):
        """
        By default our floats are generated from the normal distribution.
        If we need greater control then use from_function.
        """

        kws = kws or dict()
        kws = dict({'loc': 50, 'scale': 10}, **kws)

        return utils.from_function(iters=size,
                                   func=stats.norm.rvs,
                                   func_kws=kws,
                                   func_seed_arg='random_state',
                                   seed=seed)

    def dates(self, **kws):

        kws = kws or dict()
        start = kws.get('start')
        end = kws.get('end')
        days_ago = kws.get('days_ago')

        if not start:
            if days_ago:
                start = end - dt.timedelta(days=days_ago)
            else:
                start = end

        dates = pd.date_range(start=start, end=end)

        dates = [dt.datetime.combine(x, dt.time(0, 0, 0))
                 for x in dates.date]

        return sorted(dates)

    def tstamps(self, seed, sort=False, **kws):
        """
        Generate random timestamps associated with some dates
        (which may or may not include duplicates).

        If dates sequence is unique then only 1 tstamp per date is generated.

        So size of dates sequence determines size of tstamps array returned.
        """

        kws = kws or dict()
        for_dates = kws.get('for_dates')
        round_to = kws.get('round_to')

        prng = np.random.RandomState(seed=seed)
        # TODO: make hours pool more real-like with a higher prob to
        # occur between 9am and 6pm
        # also allow custom probabilities for minutes, perhaps to mimic
        # response to TV ads?
        # will likely need to change this with random.choice
        hours = prng.randint(low=0, high=24, size=10000)
        minutes = prng.randint(low=0, high=60, size=10000)
        seconds = prng.randint(low=0, high=60, size=10000)

        tstamps = list()

        for ix, date in enumerate(for_dates):

            # TODO: what if the for_dates has not been generated with pandas?
            if not isinstance(date, pd.tslib.NaTType):

                _prng = np.random.RandomState(seed=seed * ix)

                rand_hour = _prng.choice(hours, size=1)[0]
                rand_minute = _prng.choice(minutes, size=1)[0]
                rand_second = _prng.choice(seconds, size=1)[0]
                rand_time = dt.time(rand_hour, rand_minute, rand_second)
                rand_tstamp = dt.datetime.combine(date.date(), rand_time)

            else:
                rand_tstamp = None

            tstamps.append(rand_tstamp)

        if round_to == 'hour':

            def round_to_hour(x):
                return dt.datetime.combine(x.date(),
                                           dt.time(x.hour, 0, 0))

            return [round_to_hour(x) if x is not None else None
                    for x in tstamps]

        if sort:
            return sorted(tstamps)
        else:
            return tstamps
