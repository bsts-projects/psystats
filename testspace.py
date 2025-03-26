
import random, math
import numpy as np
import pandas as pd
from scipy import stats
from stats_project import RandomData

"""def generate_data(groups, n, distribution):
    df = pd.DataFrame()

    for group in range(groups):
        mean = random.randint(10, 100)
        sd = mean * random.uniform(0.05, 0.50)

        if distribution == "normal":
        # generate the sample based on the above values
            samples = np.random.normal(mean, sd, n)

            # round the data so it only includes whole numbers
            sample = np.round(samples).astype(int)


        # convert to a dataframe to display the data
        df[f'{group}'] = sample

    return df

df = generate_data(5, 15, "normal")
print(df)

means = []
n = 15
ss = []
for column in df: 
    means.append(round(df[column].mean(), 2))
    sum_scores = df[column].sum()
    sum_sqared_scores= (df[column].apply(lambda x: x ** 2)).sum()  
    ss_vals = sum_sqared_scores - round((sum_scores ** 2)/n, 2)
    ss.append(round(ss_vals, 2))


print(means)

print(ss)"""

df = RandomData(groups = 1, n = 15)

data = df.generate_data()
ss = df.sum_of_squares()
means = df.group_means()
sd = df.stdev()

print(data)
print(ss)
print(means)
print(sd)

df.one_sample_t_test(null = 0)

crit = df.critical_t(test = "one-sample", alpha = 0.05, tails = 1)
print(crit)