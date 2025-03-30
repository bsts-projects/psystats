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
        self.test = "" # value is set when a stats function is called
        self.alpha = self.set_alpha()
        self.tails = self.set_tails()
        self.null = self.set_null_hypothesis()
        if distribution == "normal":
            self.distribution = distribution 
        else:
            raise ValueError("only the normal distribution is currently supported")
            # TODO add the ability to generate data from other distribuions


    def set_alpha(self):
        self.alpha = random.choice([0.05, 0.01, 0.001])
        return self.alpha
        
    
    def set_tails(self):
        self.tails = random.choice([1, 2])
        return self.tails
            

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
    

    def set_null_hypothesis(self):
        # for one sample tests, sets a null hypothess between -3 to + 3 x the mean
        if  self.test in ["one-sample t-test", "z"]:
            mean = self.means[0]
            multiplier = random.uniform(-3, 3)
            self.null = round(mean * multiplier)
        else:
            self.null = 0
        return self.null


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


    def critical_t(self):
        # calculate the degrees of freedom based on the type of test used
        if self.test == "independent-samples t-test":
            degf = (self.n - 1) + (self.n - 1)
        elif self.test == "one-sample t-test" or "dependent-samples t-test":
            degf = self.n - 1
        else:
            raise ValueError("incorrect test specification - degrees of freedom")
        
        # determine the critical values for the inferential test      
        if self.tails == 1:
            if self.test in ["independent-samples t-test", "one-sample t-test", "dependent-samples t-test"]:
                crit = round(stats.t.ppf(1 - self.alpha, degf), 2)
            elif self.test == "z":
                crit = print(round(stats.norm.ppf(1 - self.alpha)), 2)
            else:
                return ValueError("improper test specification - selecting crit values")
        elif self.tails == 2:
            if self.test in ["independent-samples t-test", "one-sample t-test", "dependent-samples t-test"]: 
                crit = round(stats.t.ppf(1 - self.alpha/2, degf), 2)
            elif self.test == "z":
                crit = print(round(stats.norm.ppf(1 - (self.alpha/2)), 2))
            else:
                return ValueError("improper test specification - selecting crit values")
        else:
            return ValueError("self.tails must equal 1 or 2")
        crit_values = {"positive": crit, "negative": -crit}

        # add a direction for one-tailed tests
        if self.tails == 1:  
            direction = random.choice(["increase", "decrease"])
            crit_values["direction"] = direction
        else:
            return ValueError("tails must be 1 for directional crit values")
        
        return crit_values


    def z_test(self):
        if len(self.df.columns) > 1:
            raise Exception("Data contains more than one sample")
        elif len(self.df.columns) == 0:
            raise Exception("Dataframe error: no data columns")
        else:
            self.test = "z"
            # calculate the standard error
            # TODO double check the work here to make sure it is accurate
            sd = self.df['0'].std(ddof = 0)
            n = len(self.df['0'])
            sem = round(sd/(round(math.sqrt(n),2)),2)
            z_obt = round((self.means[0] - self.null) / sem, 2)

            # TODO add a way to determine environment so output can display in terminal or notebook
            # print calculations for the standard error
            display(Markdown("Calculating the standard error..."))
            display(Markdown(f"$\\sigma_M = \\frac{{\\sigma}}{{\\sqrt{{N}}}}$"))
            display(Markdown(f"$\\sigma_M = \\frac{{{sd}}}{{\\sqrt{n}}}$"))
            display(Markdown(f"$\\sigma_M = \\frac{{{sd}}}{{{round(math.sqrt(n),2)}}}$"))
            display(Markdown(f"$\\sigma_M = {{{sem}}}$"))
            print() # blank space
            # print the caluclations for z_obt
            display(Markdown("calculating $z_{{obt}}$..."))
            display(Markdown(f"$z_{{obt}} = {{\\frac{{M - \\mu}}{{\\sigma_M}}}}$"))
            display(Markdown(f"$z_{{obt}} = \\frac{{{self.means[0]} - {self.null}}}{{{sem}}}$"))
            display(Markdown(f"$z_{{obt}} = \\frac{{{self.means[0] - self.null}}}{{{sem}}}$"))
            display(Markdown(f"$z_{{obt}} = {{{z_obt}}}$"))

            return z_obt  


    def one_sample_t_test(self):
        if len(self.df.columns) > 1:
            raise Exception("Data contains more than one sample")
        elif len(self.df.columns) == 0:
            raise Exception("Dataframe error: no data columns")
        else:
            self.test = "one-sample t-test"   
            # calculate the standard error
            sem = round(math.sqrt(round((self.var[0]/self.n),2)),2)
            t_obt = round((self.means[0] - self.null) / sem, 2)

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
            display(Markdown(f"$t_{{obt}} = \\frac{{{self.means[0]} - {self.null}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = \\frac{{{self.means[0] - self.null}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = {{{t_obt}}}$"))

            return t_obt         


    def independent_samples_t_test(self):
        if len(self.df.columns) == 1 or len(self.df.columns) > 2:
            raise ValueError("Data does not contain two samples")
        elif len(self.df.columns) == 0:
            raise ValueError("Dataframe error: no data columns")
        else: 
            self.test = "independent-samples t-test"           
            # primary calculations
            pooled_var = round(((self.ss[0] + self.ss[1]) / ((self.n - 1) + (self.n - 1))), 2)
            sem = round(math.sqrt((round((pooled_var/self.n),2))+(round((pooled_var/self.n),2))),2)
            t_obt = round(((self.means[0] - self.means[1]) - self.null) / sem, 2)

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
            display(Markdown(f"$t_{{obt}} = \\frac{{({self.means[0]} - {self.means[1]}) - {{{self.null}}}}}{{{sem}}}$")) 
            display(Markdown(f"$t_{{obt}} = \\frac{{{round(self.means[0] - self.means[1] - self.null, 2)}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = {{{t_obt}}}$"))

            return t_obt
        

    def dependent_samples_t_test(self):
        if len(self.df.columns) == 1 or len(self.df.columns) > 2:
            raise ValueError("Data does not contain two samples")
        elif len(self.df.columns) == 0:
            raise ValueError("Dataframe error: no data columns")
        else:
            self.test = "dependent-samples t-test"     
            # primary calculations
            # need to gather the difference scores
            self.df['D'] = self.df['1'] - self.df['0']
            
            # print the dataframe with the difference scores
            display(Markdown("Calculating the difference scores $D = X_1 - X_0$"))
            print(self.df.to_string(index = False))
            
            # Calculate the Mean of the Difference Scores
            sum_d = self.df['D'].sum()
            n = len(self.df['D'])
            mean_d = round(sum_d/n, 2)
            display(Markdown("Calculating the Mean of the Difference Scores..."))
            display(Markdown("$M_D = \\frac{{\\Sigma D}}{{n}}$"))
            display(Markdown(f"$M_D = \\frac{{{sum_d}}}{{{n}}}$"))
            display(Markdown(f"$M_D = {{{mean_d}}}$"))

            # calculate the SS for the difference scores
            self.df['D^2'] = self.df['D'].apply(lambda x: x ** 2)
            sum_sqared_scores = self.df['D^2'].sum()
            ss = round(sum_sqared_scores - round((sum_d ** 2)/n, 2), 2)
            # print the dataframe with the squared difference scores
            display(Markdown("Calculating the sum of the squared deviations..."))
            print(self.df.to_string(index = False))
            display(Markdown("$ SS_D = \\Sigma D^2 - \\frac{{(\\Sigma D)^2}}{{n}}$"))
            display(Markdown(f"$ SS_D = {{{sum_sqared_scores}}} - \\frac{{{sum_d ** 2}}}{{{n}}}$"))
            display(Markdown(f"$ SS_D = {{{sum_sqared_scores}}} - {{{round((sum_d ** 2)/n, 2)}}}$"))
            display(Markdown(f"$ SS_D = {{{ss}}}$"))

            # calculate the variance    
            variance = round(ss / (n - 1), 2)
            display(Markdown("$ s^2 = \\frac{{SS_D}}{{n}}$")) 
            display(Markdown(f"$ s^2 = \\frac{{{ss}}}{{{n}}}$"))   
            display(Markdown(f"$ s^2 = \\frac{{{round(ss/n, 2)}}} $"))
            display(Markdown(f"$ s^2 = {{{variance}}}$"))  

            # Calculate the estimated standard error
            sem = round(math.sqrt(variance/n), 2)
            display(Markdown("Calculating the estimated standard error..."))
            display(Markdown("$ s_{M_D} = \\sqrt{{\\frac{{s^2}}{{n}}}}$"))
            display(Markdown(f"$ s_{{M_D}} = \\sqrt{{\\frac{{{variance}}}{{{n}}}}}$"))
            display(Markdown(f"$ s_{{M_D}} = \\sqrt{{{round(variance/n, 2)}}}$"))
            display(Markdown(f"$ s_{{M_D}} = {{{sem}}}$"))

            # caclulate the t-statistic
            t_obt = round((mean_d - self.null) / sem, 2)
            display(Markdown("calculating $t_{{obt}}$..."))
            display(Markdown("$t_{{obt}} = {{\\frac{{M_D - \\mu_D}}{{s_{M_D}}}}}$"))
            display(Markdown(f"$t_{{obt}} = \\frac{{{mean_d} - {self.null}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = \\frac{{{mean_d - self.null}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = {{{t_obt}}}$"))

            return t_obt
