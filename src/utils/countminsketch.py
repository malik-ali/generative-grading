# -*- coding: utf-8 -*-
import hashlib
import array
import numpy as np

class CountMinSketch(object):
    """
    A class for counting hashable items using the Count-Min—Sketch algorithm.
    Serves as an efficient, approximate version of itertools.Counter
    
    Parameters
     - `num_buckets` the size of each hash table
     - `num_hashes` the number of hashes
    """

    def __init__(self, num_buckets=10000, num_hashes=20):
        """ `m` is the size of the hash tables, larger implies smaller
        overestimation. `d` the amount of hash tables, larger implies lower
        probability of overestimation.
        """
        self.num_buckets = num_buckets
        self.num_hashes = num_hashes
        self.n = 0
        self.tables = np.zeros(shape=(num_hashes, num_buckets), dtype=int)
    
    def _hash(self, x):
        md5 = hashlib.md5(str(hash(x)).encode('utf-8'))
        for i in range(self.num_hashes):
            md5.update(str(i).encode('utf-8'))
            yield int(md5.hexdigest(), 16) % self.num_buckets

    def add(self, x):
        """
        Count element `x` as if had appeared `value` times.
        By default `value=1` so:
            sketch.add(x)
        Effectively counts `x` as occurring once.
        """
        self.n += 1
        for table, i in zip(self.tables, self._hash(x)):
            table[i] += 1

    def query(self, x):
        """
        Return an estimation of the amount of times `x` has ocurred.
        The returned value always overestimates the real value.
        """
        return min(table[i] for table, i in zip(self.tables, self._hash(x)))

    def __getitem__(self, x):
        """
        A convenience method to call `query`.
        """
        return self.query(x)

    def __len__(self):
        """
        The amount of things counted. Takes into account that the `value`
        argument of `add` might be different from 1.
        """
        return self.n