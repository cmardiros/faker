import logging
import copy
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
from sklearn.model_selection import ParameterGrid


def from_function(func,
                  func_kws=None,
                  func_seed_arg=None,
                  iters=100,
                  seed=12345):
    """
    Create a numpy array with `iters` rows where each row is the result of a
    function.

    Parameters:
    -----------
        func_seed_arg: name of argument to pass to func to set a seed
        (for scipy.rvs this is random_state)

    Examples:

    # generate 10 numbers from the standard normal distribution
    # and repeat the process 5 times
    # returns: np array with 5 rows, each row represents a
    # random sample of 10 values from the standard normal
    from_function(func=stats.norm.rvs,
                  func_kws=dict(loc=0, scale=1, size=10),
                  iters=5,
                  func_seed_arg='random_state',
                  seed=12345)


    # throw a dice 10 times and repeat process 5 times
    # returns: np array with 5 rows, each row represents 10 throws
    dice = np.arange(1, 7, 1)
    from_function(func=np.random.choice,
                  func_kws=dict(a=dice, size=10, replace=True),
                  iters=5)

    """

    func_kws = func_kws or dict()

    seeds = get_seeds(size=iters, initial_seed=seed)

    samples = list()
    for ix in range(iters):
        if func_seed_arg and seed:
            func_kws[func_seed_arg] = seeds[ix]
        samples.append(func(**func_kws))

    return np.array(samples)


def stacked_from_function(param_grid,
                          apply_func=None,
                          apply_func_kws=None,
                          col_names=None,
                          make_cats=None,
                          **kws):
    """
    Generate some data using a function which accepts some parameters and
    create a stacked pandas dataframe which can be plotted as facets to
    compare and contrast.

    Parameters:
    -----------

        apply_func: function to apply to each generated row, so each row
        effectively becomes the result of that function

        apply_func_kws: arguments to pass to apply_func

        col_names: column names for the resulting dataframe

        make_cats: list, optional
            These columns will be turned into pandas categories
    """

    param_grid = sklearn.model_selection.ParameterGrid(param_grid)

    apply_func_kws = apply_func_kws or dict()
    col_names = col_names or list()
    make_cats = make_cats or list()

    dfs = list()

    for params in param_grid:

        data = from_function(func_kws=copy.deepcopy(params), **kws)

        if apply_func:
            data = apply_func(data, axis=1, **apply_func_kws)

        df = pd.DataFrame(data)

        if col_names:
            df.rename(columns={c: col_names[i]
                               for i, c in enumerate(df.columns)},
                      inplace=True)
        else:
            df.rename(columns={c: 'col_{}'.format(c) for c in df.columns},
                      inplace=True)

        # create categorical columns for each parameter passed to function
        # as we will use these for facetting when we plot the data
        for k, v in params.items():
            df[k] = v

        dfs.append(df)

    dfm = pd.concat(dfs)

    for cat in make_cats:
        dfm[cat] = dfm[cat].astype('category')

    return dfm


def generate_probs(size,
                   dist_kws,
                   seed=12345):
    """
    Generate a sequence of `size` probabilities that follow a certain
    scipy distribution. This is useful for generating fake data by
    using this in conjunction with np.random.choice.

    Example:

    # data to resample from with our new probabilities
    pool = np.arange(1, 101, 1)
    n_pool = pool.shape[0]

    # size of the faked dataset
    n_resample = n_pool * 10

    probs = generators.generate_probs(
        size=n_pool,
        dist_kws={'family': 'expon', 'loc': 0, 'scale': 1},
        seed=12345)

    sns.distplot(probs)

    rands = np.random.choice(pool,
                             size=n_resample,
                             replace=True,
                             p=probs)
    sns.distplot(rands)
    """

    dist_family = dist_kws.pop('family')

    dist_factory = getattr(stats, dist_family)

    mydist = dist_factory(**dist_kws)

    data = mydist.rvs(size=size, random_state=seed)

    # normalise so that they add up to 1
    probs = data / data.sum()
    probs /= probs.sum()

    # sort and revert so we have highest probs first
    probs = np.flip(np.sort(probs), axis=0)

    return probs


def get_seeds(size, initial_seed=12345):
    """
    Generate a reproducible list of integers to be used as seeds.
    """

    prng = np.random.RandomState(seed=initial_seed)
    ints = prng.randint(low=100, high=1000, size=100000)

    return prng.choice(a=ints, size=size, replace=False)
