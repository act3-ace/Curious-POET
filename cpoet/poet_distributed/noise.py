# The following code is modified from openai/evolution-strategies-starter
# (https://github.com/openai/evolution-strategies-starter)
# under the MIT License.

# Modifications Copyright (c) 2020 Uber Technologies, Inc.


# imports
import logging
import numpy as np
from multiprocessing.sharedctypes import RawArray
from ctypes import c_float as cF

# initialize logger
logger = logging.getLogger(__name__)

# File constants
DEBUG = False
SEED = 42


class SharedNoiseTable(object):
    def __init__(self):

        # amount of random numbers to draw
        #  1gb (or 4mb) of 32-bit numbers
        #  actually sample 2x that, as 64-bit numbers, then reduce precision for storage
        self.noiseLen = 250000000 if not DEBUG else 1000000

        # log creation of noise
        logger.info(f'Sampling {self.noiseLen} random numbers with seed {SEED}')

        # create shared object
        #  this is not thread safe, but that's fine because we treat noise objects as read-only
        self._shared_mem = RawArray(typecode_or_type=cF, size_or_initializer=self.noiseLen)

        # cast as numpy object for easy access
        self.noise = np.ctypeslib.as_array(self._shared_mem)

        # fill with standard normal
        #  64-bit to 32-bit conversion done when pulling
        #  current default prng is PCG64, we're using more robust DXSM version
        prng = np.random.Generator(np.random.PCG64DXSM(seed=SEED))
        prng.standard_normal(size=self.noiseLen, dtype=cF, out=self.noise)

        # log success
        logger.info(f'Sampled {self.noiseLen * 4} bytes')

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.integers(self.noiseLen - dim, endpoint=True)
