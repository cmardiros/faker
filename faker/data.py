import logging
import pandas as pd
from faker import column


class FakeDataGenerator(object):
    """
    Generate a df of fake data.
    """

    def __init__(self,
                 rows=100):
        self.rows = rows

    def generate(self,
                 cols=None,
                 seed=12345):

        cols = cols or list()
        data = list()

        for ix, col_kws in enumerate(cols):
            name = col_kws.pop('name')
            dtype = col_kws.pop('dtype')

            generator = column.FakeColumnGenerator(
                rows=self.rows,
                nan_ratio=col_kws.pop('nan_ratio', 0),
                zero_ratio=col_kws.pop('zero_ratio', 0))

            values = generator.generate(seed=seed * (ix + 1), **col_kws)

            data.append((name, values, dtype))

        headers = [header for header, values, dtype in data]
        records = {header: values for header, values, dtype in data}
        dtypes = {header: dtype for header, values, dtype in data}

        df = pd.DataFrame.from_records(data=records,
                                       columns=headers,
                                       coerce_float=True)

        for col, dtype in dtypes.items():
            df[col] = df[col].astype(dtype)

        return df
