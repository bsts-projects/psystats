import numpy as np
import pandas as pd
import random, math
from scipy import stats
from IPython.display import Markdown, display

class RandomData():
    def __init__(self, groups = 1, n = 10, distribution = "normal"):
        self.groups = groups
        self.n = n
        self.distribution = distribution # placeholder to add the ability to generate data from other distributions
        
            
    def generate_data(self):
        self.df = pd.DataFrame()
        # create data for each group and add it to the dataframe
        for group in range(self.groups):
            mean = random.randint(10, 100)
            sd = mean * random.uniform(0.05, 0.50)

            # generate the sample based on the above values
            samples = np.random.normal(mean, sd, self.n)

            # round the data so it only includes whole numbers
            sample = np.round(samples).astype(int)

            # convert to a dataframe to display the data
            self.df[f'{group}'] = sample
        return self.df
    

    def sum_of_squares(self):  # calculating the values presented in the problem.
        ss = []
        for column in self.df: 
            sum_scores = self.df[column].sum()
            sum_sqared_scores= (self.df[column].apply(lambda x: x ** 2)).sum()  
            ss_vals = sum_sqared_scores - round((sum_scores ** 2)/self.n, 2)
            ss.append(round(ss_vals, 2))
        return ss


    def group_means(self):
        means = []
        for column in self.df:
            means.append(round(self.df[column].mean(), 2))  
        return means
    

    def variance(self):
        vars = []
        for column in self.df:
            vars.append(round(self.df[column].var(ddof = 1), 2))
        return vars


    def stdev(self):
        stdevs = []
        for column in self.df:
            stdevs.append(round(self.df[column].std(ddof = 1), 2))
        return stdevs


    def one_sample_t_test(self, null: int):
        if len(self.df.columns) > 1:
            raise Exception("Data contains more than one sample")
        elif len(self.df.columns) == 0:
            raise Exception("Dataframe error: no data columns")
        else:            
            # calculate the standard error
            var = self.df.variance()
            sem = round(math.sqrt(round(var/self.n),2),2)
            mean = self.df.group_means()
            mu = null
            t_obt = round((mean - mu) / sem, 2)

            print("calculating the standard error...")
            display(Markdown("$s_M = \\sqrt{{\\frac{{s^2}}{{n}}}}$"))
            display(Markdown(f"$s_M = \\sqrt{{\\frac{{{var}}}{{{self.n}}}}}$"))
            display(Markdown(f"$s_M = \\sqrt{{{round((var/self.n),2)}}}$"))
            display(Markdown(f"$s_M = {{{sem}}}$"))

            display(Markdown("calculating $t_{(obt}}$..."))
            display(Markdown("$t_{{obt}} = \\frac{{M - \\mu}}{{s_M}}"))
            display(Markdown(f"$t_{{obt}} = \\frac{{{mean} - {mu}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = \\frac{{{mean - mu}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = {{{t_obt}}}"))


            
            



