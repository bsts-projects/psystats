
import random, math
import numpy as np
import pandas as pd
from scipy import stats
from stats_project import RandomData



data = RandomData(groups = 2, n = 10)


data.independent_samples_t_test()