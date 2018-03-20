import logging
import string
import numpy as np


class FakeStringValueGenerator(object):
    """
    Generate a single string value.
    """

    char_mapping = {
        'tab': '\t',
        'escaped_tab': '\\t',
        'comma': ',',
        'escaped_comma': '\,',
        'newline': '\n',
        'escaped_newline': '\\n',
        'carriage': '\r',
        'escaped_carriage': '\\r',
        'newlinecarriage': '\n\r',
        'escaped_newlinecarriage': '\\n\\r',
    }

    hosts = ['@gmail.com',
             '@yahoo.com',
             '@hotmail.com',
             '@blueyonder.com']

    def __init__(self,
                 length,
                 digits=False,
                 letters=False,
                 tab=False,
                 escaped_tab=False,
                 comma=False,
                 escaped_comma=False,
                 newline=False,
                 escaped_newline=False,
                 carriage=False,
                 escaped_carriage=False,
                 newlinecarriage=False,
                 escaped_newlinecarriage=False,
                 prefix=None,
                 suffix=None,
                 email=False):

        self.length = length
        self.digits = digits
        self.letters = letters
        self.tab = tab
        self.escaped_tab = escaped_tab
        self.comma = comma
        self.escaped_comma = escaped_comma
        self.newline = newline
        self.escaped_newline = escaped_newline
        self.carriage = carriage
        self.escaped_carriage = escaped_carriage
        self.newlinecarriage = newlinecarriage
        self.escaped_newlinecarriage = escaped_newlinecarriage
        self.email = email
        self.suffix = suffix
        self.prefix = prefix

    def generate(self, seed):

        prng = np.random.RandomState(seed=seed)

        # get random characters from the pool
        rands = list(prng.choice(self.pool(), size=self.length))

        rands = self.insert_from_mapping(rands, seed=seed)

        if self.email:
            rands = self.make_email(rands, seed=seed)

        if self.suffix:
            rands = rands + [self.suffix]

        if self.prefix:
            rands = [self.prefix] + rands

        return ''.join(rands)

    def pool(self):

        pool = list()

        if self.digits:
            pool += [c for c in string.digits]

        if self.letters:
            pool += [c for c in string.ascii_uppercase]

        return pool

    def insert_from_mapping(self, rands, seed):

        indices = np.arange(0, len(rands), 1)

        for ix, key in enumerate(self.char_mapping.keys()):

            if getattr(self, key):

                # create a prng for this iteration only
                _prng = np.random.RandomState(seed=seed * (ix + 1))

                # replace a random character with mapping
                loc = _prng.choice(a=indices, size=1, replace=False)
                rands[loc] = self.char_mapping[key]

        return rands

    def make_email(self, rands, seed):

        prng = np.random.RandomState(seed=seed)

        host = list(prng.choice(self.hosts, size=1))
        rands += host

        return rands
