import numpy as np


class FakeDataInspector(object):

    def __init__(self):
        pass

    def pii_mapping(self, df):
        """
        Inspect a PII mapping for dupes and many-to-many relationships.

        i.e if the pii_mapping includes a column of cookie IDs and user IDs
        check whether we have:
        - cookie IDs mapped to multiple users
        - user IDs mapped to multiple cookies
        """

        # general frequency counts
        for col in df.columns:
            series = df[col]
            print("\n***col: {}".format(col))
            print("\n>>describe:\n{}".format(series.describe()))
            freqs = series.value_counts(dropna=False)
            print("\n>>freqs:\n{}".format(freqs))
            freq_ratios = (series.value_counts(dropna=False) /
                           series.shape[0])
            print("\n>>freq ratios:\n{}".format(freq_ratios))

        # create a many-to-many mapping
        # e.g. which cookie_ids have multiple matches in the user_id column
        many_matches = dict()

        if len(df.columns) > 1:

            for col in df.columns:
                # e.g. cookie_id
                many_matches[col] = dict()

                # e.g. ['user_id', 'email_address']
                other_cols = [c for c in df.columns if c != col]

                for other_col in other_cols:
                    # discard nulls from our counts
                    filtered_df = df[~df[other_col].isnull()]
                    gb = filtered_df.groupby([col]).agg([len])[[other_col]]
                    gb.columns = gb.columns.map(' '.join)
                    gb.reset_index(inplace=True)
                    gb = gb.sort_values(by='{} len'.format(other_col),
                                        ascending=False)
                    gb = gb[gb['{} len'.format(other_col)] > 1][col]
                    if gb.shape[0] > 0:
                        print("{} with multiple matches in {}:\n{}".format(
                            col, other_col, gb))

                    # get records that have multiple matches
                    many_matches[col][other_col] = gb.values

        # inspect many to many
        for col, other_col in many_matches.items():
            print("\n***col: {}".format(col))
            for other_col in other_col.keys():
                matches = many_matches[col][other_col]

                _df = df[df[col].isin(matches)]
                _df = _df.sort_values(by=col)
                if _df.shape[0] > 0:
                    print("\n{} has many matches in {}:\n{}".format(
                        col, other_col, _df))

        # filter mapping for dupes only
        dupes = dict()
        for col, other_col in many_matches.items():
            dupes[col] = list()
            for other_col in other_col.keys():
                # save for later
                matches = many_matches[col][other_col]
                dupes[col] += list(matches)
                dupes[col] = list(set(dupes[col]))

        matches = np.zeros(shape=(df.shape[0], ))
        for col in dupes.keys():
            print("\n***col: {}".format(col))
            duplicates = dupes[col]
            print("duplicates: {}".format(duplicates))
            filtered = df[col].isin(duplicates)
            print("filtered.sum()", filtered.sum())
            matches += filtered

        return df.loc[matches > 0].sort_values(by=list(df.columns))
