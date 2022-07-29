# batchgauss

Batch sample from N multivariate (M-dimensional) Gaussians using `numpy`.

## Usage

There is only one function (defined in `__init__.py`), `sample`.

Here is an example of usage:
```python
import numpy as np
from batchgauss import sample

N = 1000
M = 2

# set up means and covariances
means = np.zeros((N, M))
covs = np.zeros((N, M, M))
means[:500, 1] = -5
means[500:, 1] = 5
covs[:, 0, 0] = 1
covs[:, 1, 1] = 1
covs[:500, 0, 1] = 0.5
covs[:500, 1, 0] = 0.5
covs[500:, 0, 1] = -0.5
covs[500:, 1, 0] = -0.5

# draw samples
y = sample(means, covs)
```
See the function docstring for more details.

## Prerequisites

The only prerequisite is `numpy`. My code was developed and tested using version 1.21.5, but earlier/later versions likely to work also.
