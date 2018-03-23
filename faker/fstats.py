import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.model_selection import ParameterGrid


def logistic(X, betas, alphas=None, noise=None):
    """
    Simulate data from a logistic regression model with one variable.

    Parameters:
    -----------
        X: 1-d array, required
            Values for our X predictor variable

        betas: iterable of 2 values, required
            Shape parameter/slope
            betas[0] = coeff values for X predictor variable
            betas[1] = coeff values for intercept (alpha)
            Typically if intercept is introduced into the model beta[1] = 1

        alphas: 1-d array of length X.shape[0], optional
            Intercept/bias values for X (shift parameter)
            Typically this is a constant value across all values of X
            e.g. np.full(shape=X.shape, fill_value=1)

        noise: 1-d array of length X.shape[0], optional
            Noise value to add to each value of X
            e.g. stats.norm.rvs(loc=0, scale=1, size=X.shape[0])

    """

    # add the intercept values to X
    # effectively shifts our entire X array by the value of alpha
    # creates an array with 2 rows, each row of length X.shape[0]
    if alphas is not None:
        X = np.vstack([X, alphas])

    # compute predicted log-odds for every value of X
    # logodds = coeff * x + intercept
    # x1*beta_x + intercept1*beta_intercept
    # + x2*beta_x + intercept2*beta_intercept ...
    logodds = X.T.dot(betas)

    # add noise if exists
    # TODO: why is noise added to logodds?
    if noise is not None:
        logodds = logodds + noise

    # convert log-odds to odds (aka odds ratio)
    # -- v important for interpretation
    # indicator of the change in odds resulting from 1 unit change in predictor
    # If > 1 then = as the predictor increases, the odds of the outcome occuring increases
    # If < 1 then = as the predictor decreases, the odds of the outcome occuring decreases
    # TODO: should logodds be negative here?
    odds = np.exp(logodds)

    # convert odds to probability
    prob = odds / (1 + odds)

    # back to odds
    # odds = probability of event / probability of non-event
    odds = prob / (1 - prob)

    return prob


def logistic_grid(grid):
    """
    Pass a grid of parameters to simulate logistic regression data from
    and union all resulting dataframes.

    Example:
    shape_grid = {

        # list of (X, intercept=1)
        'X_alphas': [
            (X, np.full(shape=X.shape, fill_value=1)),
        ],

        # beta_x and beta_intercept
        'betas': [
            (1, 1)
        ],

        'noise': [
            stats.norm.rvs(loc=0, scale=1, size=X.shape[0])
        ]
    }
    """

    grid = sklearn.model_selection.ParameterGrid(grid)

    dfs = list()

    for ix, params in enumerate(grid):

        # first element is X
        X = params['X_alphas'][0]

        # second element is associated intercept array
        alphas = params['X_alphas'][1]

        betas = params['betas']

        noise = params.get('noise')

        y_prob = logistic(X=X,
                          betas=betas,
                          alphas=alphas,
                          noise=noise)

        df = pd.DataFrame(np.vstack([X, alphas]).T, columns=['x', 'intercept'])
        df['beta_x'] = betas[0]
        df['beta_intercept'] = betas[1]
        df['noise'] = noise

        # probabilities
        df['y_prob'] = y_prob

        # NOTE:original odds and log odds calculated in the logistic
        # match df['y_odds'] and df['y_logodds'] respectively

        # log-odds
        df['y_logodds'] = scipy.special.logit(df['y_prob'])

        # odds aka odds-ratio
        df['y_odds'] = np.exp(df['y_logodds'])
        # ^^ is equivalent to
        # df['y_odds'] = df['y_prob'] / (1 - df['y_prob'])

        # discretized
        df['y_discrete'] = (df['y_prob'] > 0.5) * 1

        # indices
        df['param_ix'] = ix + 1

        dfs.append(df)

    dfm = pd.concat(dfs)

    return dfm
