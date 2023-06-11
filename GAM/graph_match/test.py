import io
import os
import re
import collections
import numpy as np
from config import get_default_config
import torch

a = torch.Tensor([1,2,3,4])
a=torch.stack([a])
print(a.T)