import numpy as np
import pandas as pd
import random, math
from scipy import stats
from IPython.display import Markdown, display

class RandomData():
    def __init__(self, groups = 1, n = 10, distribution = "normal"):
        self.groups = groups
        self.n = n
        self.df = self.generate_data()
        self.ss = self.sum_of_squares()
        self.means = self.group_means()
        self.var = self.variance()
        self.std = self.stdev()
        if distribution == "normal":
            self.distribution = distribution 
        else:
            raise ValueError("only the normal distribution is currently supported")
            # TODO add the ability to generate data from other distribuions
            
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


    def critical_t(self, test, alpha = 0.05, tails = 2):
        if test == "independent-samples":
            degf = (self.n - 1) + (self.n - 1)
        elif test == "one-sample" or "dependent-samples":
            degf = self.n - 1
        else:
            raise ValueError("Test options include:  'independent-samples', 'one-sample' or 'dependent-samples'")
        
        crit_values = {}
        if tails == 1:
            t_crit = stats.t.ppf(1 - alpha, degf)
            crit_values["t_crit_pos"] = round(t_crit, 3)
            crit_values["t_crit_neg"] = round(-t_crit, 3)
        elif tails == 2:
            t_crit_upper = stats.t.ppf(1 - alpha/2, degf)
            t_crit_lower = stats.t.ppf(alpha/2, degf)
            crit_values["t_crit_upper"] = round(t_crit_upper, 3)
            crit_values["t_crit_lower"] = round(t_crit_lower, 3)
        else:
            return ValueError("tails must be 1 or 2")

        return crit_values


    def one_sample_t_test(self, null: int):
        if len(self.df.columns) > 1:
            raise Exception("Data contains more than one sample")
        elif len(self.df.columns) == 0:
            raise Exception("Dataframe error: no data columns")
        else:            
            # calculate the standard error
            #var = self.variance()
            sem = round(math.sqrt(round((self.var[0]/self.n),2)),2)
            #mean = self.group_means()
            mu = null
            t_obt = round((self.means[0] - mu) / sem, 2)

            # print the caluclations for the standard error
            # TODO add a way to determine environment so output can display in terminal or notebook
            print("calculating the standard error...")
            display(Markdown("$s_M = \\sqrt{{\\frac{{s^2}}{{n}}}}$"))
            display(Markdown(f"$s_M = \\sqrt{{\\frac{{{self.var[0]}}}{{{self.n}}}}}$"))
            display(Markdown(f"$s_M = \\sqrt{{{round((self.var[0]/self.n),2)}}}$"))
            display(Markdown(f"$s_M = {{{sem}}}$"))
            print() # blank space
            # print the caluclations for t_obt
            display(Markdown("calculating $t_{{obt}}$..."))
            display(Markdown(f"$t_{{obt}} = {{\\frac{{M - \\mu}}{{s_M}}}}$"))
            display(Markdown(f"$t_{{obt}} = \\frac{{{self.means[0]} - {mu}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = \\frac{{{self.means[0] - mu}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = {{{t_obt}}}$"))

            return t_obt         


    def independent_samples_t_test(self, null: int):
        if len(self.df.columns) == 1 or len(self.df.columns) > 2:
            raise ValueError("Data does not contain two samples")
        elif len(self.df.columns) == 0:
            raise ValueError("Dataframe error: no data columns")
        else:            
            # primary calculations
            pooled_var = round(((self.ss[0] + self.ss[1]) / ((self.n - 1) + (self.n - 1))), 2)
            sem = round(math.sqrt((round((pooled_var/self.n),2))+(round((pooled_var/self.n),2))),2)
            t_obt = round(((self.means[0] - self.means[1]) - null) / sem, 2)

            # TODO adapt to display in the terminal or a notebook
            # display the caluclations for the pooled variance
            print("calculatig the pooled variance...")
            display(Markdown("$s_p^2 = {{\\frac{{SS_1 + SS_2}}{{df_1 + df_2}}}}$"))
            display(Markdown(f"$s_p^2 = {{\\frac{{{self.ss[0]} + {self.ss[1]}}}{{{self.n - 1} + {self.n - 1}}}}}$"))
            display(Markdown(f"$s_p^2 = \\frac{{{round(self.ss[0] + self.ss[1],2)}}}{{{(self.n - 1) + (self.n - 1)}}}$"))
            display(Markdown(f"$s_p^2 = {{{pooled_var}}}$"))
            # display the calculations for the estimated standard error
            print("calculating the estimated standard error of the difference between means...")
            display(Markdown("$s_{{(M_1 - M_2)}} = \\sqrt{{\\frac{{s_p^2}}{{n_1}} + \\frac{{s_p^2}}{{n_1}}}}$"))
            display(Markdown(f"$s_{{(M_1 - M_2)}} = \\sqrt{{\\frac{{{pooled_var}}}{{{self.n}}} + \\frac{{{pooled_var}}}{{{self.n}}}}}$"))
            display(Markdown(f"$s_{{(M_1 - M_2)}} = \\sqrt{{{round(pooled_var/self.n, 2)} + {round(pooled_var/self.n, 2)}}}$"))
            display(Markdown(f"$s_{{(M_1 - M_2)}} = \\sqrt{{{round(pooled_var/self.n, 2) + round(pooled_var/self.n, 2)}}}$"))
            display(Markdown(f"$s_{{(M_1 - M_2)}} = {{{sem}}}$"))
            # display the caluclations for t_obt
            display(Markdown("calculating $t_{{obt}}$..."))
            display(Markdown(f"$t_{{obt}} = {{\\frac{{(M_1 - M_2) - (\\mu_1 - \\mu_2)}}{{s_{{(M_1 - M_2)}}}}}}$"))
            display(Markdown(f"$t_{{obt}} = \\frac{{({self.means[0]} - {self.means[1]}) - {{{null}}}}}{{{sem}}}$")) 
            display(Markdown(f"$t_{{obt}} = \\frac{{{round(self.means[0] - self.means[1] - null, 2)}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = {{{t_obt}}}$"))

            return t_obt
        

    def dependent_samples_t_test(self):
        if len(self.df.columns) == 1 or len(self.df.columns) > 2:
            raise ValueError("Data does not contain two samples")
        elif len(self.df.columns) == 0:
            raise ValueError("Dataframe error: no data columns")
        else:            
            # primary calculations
            # need to gather the difference scores
            pass
