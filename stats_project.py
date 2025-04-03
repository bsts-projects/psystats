import numpy as np
import pandas as pd
import random, math
from scipy import stats
from IPython.display import Markdown, display

class RandomData():
    def __init__(self, groups = 1, n = 10, distribution = "normal"):
        self.groups = groups
        self.n = n # TODO add option for unequal sample sizes.
        self.df = self.generate_data()
        self.ss = self.sum_of_squares()
        self.means = self.group_means()
        self.var = self.variance()
        self.std = self.stdev()
        self.test = "" # value is set when a stats function is called
        self.alpha = self.set_alpha()
        self.tails = self.set_tails()
        self.null = int
        self.obt = float
        self.effect_size = float
        self.crit_values = {}; dict
        self.significance = bool
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
    

    def generate_question(self):
        # determine the test type
        if self.test == "z":
            display(Markdown(f"Given the following data, does $Group_0$ significantly differ from ${{{self.null}}}$?  Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$"))
            display(Markdown(f"$M_0 = {{{self.means[0]}}}$"))
            display(Markdown(f"$ {{\\sigma_0}} = {{{round(self.df['0'].std(ddof = 0), 2)}}}$"))
            display(Markdown(f"$ n = {{{len(self.df['0'])}}}$"))
        elif self.test == "one-sample t-test":
            display(Markdown(f"Given the following data, does $Group_0$ significantly differ from ${{{self.null}}}$?  Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$"))
            display(Markdown(f"$M_0 = {{{self.means[0]}}}$"))
            display(Markdown(f"$s^2 = {{{self.var[0]}}}$"))
            display(Markdown(f"$ n = {{{len(self.df['0'])}}}$"))
        elif self.test == "independent-samples t-test":
            display(Markdown(f"Given the following between-subjects data, does $Group_0$ significantly differ from $Group_1$?  Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$"))
            display(Markdown(f"$M_0 = {{{self.means[0]}}}, M_1 = {{{self.means[1]}}}$"))
            display(Markdown(f"$SS_0 = {{{self.ss[0]}}}, SS_1 = {{{self.ss[1]}}}$"))
            display(Markdown(f"$ n_0 = {{{len(self.df['0'])}}}, n_1 = {{{len(self.df['1'])}}}$"))
        elif self.test == "dependent-samples t-test":
            display(Markdown(f"Given the following within-subjects data, does $M_0$ significantly differ from $M_1$?  Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$"))
            display(Markdown(f"$M_0 = {{{self.means[0]}}}, M_1 = {{{self.means[1]}}}$"))
            display(Markdown(f"$ n = {{{len(self.df['0'])}}}$"))
        else:
            return ValueError("test-type specification error in question geneneration")
        
        print(self.df)
        

    def final_decision(self):
        if self.tails == 2:
            if self.obt > self.crit_values["positive"] or self.obt < self.crit_values["negative"]:
                self.significance = True
            else:
                self.significance = False
        elif self.tails == 1:
            if self.crit_values["direction"] == "increase" and self.obt > self.crit_values["positive"]:
                self.significance = True
            elif self.crit_values["direction"] == "decrease" and self.obt < self.crit_values["negative"]:
                self.significance = True
            else:
                self.significance = False
        else:
            return ValueError("error in tails specification for final decision")
        
        return self.significance 
    

    def write_result(self):
        # TODO add more elaborate functionality for the results
        if self.test in ["independent-samples t-test", "one-sample t-test", "dependent-samples t-test"]:
            # print the critical value for the test
            if self.tails == 2:
                display(Markdown(f"$t_{{crit}} = \\pm{{{self.crit_values['positive']}}}, \\alpha_{{two-tailed}} = {{{self.alpha}}}, df = {{{self.crit_values['degf']}}}$"))
            elif self.tails == 1 and self.crit_values["direction"] == "increase":
                display(Markdown(f"$t_{{crit}} = +{{{self.crit_values['positive']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}, df = {{{self.crit_values['degf']}}}$"))
            elif self.tails == 1 and self.crit_values["direction"] == "decrease":
                display(Markdown(f"$t_{{crit}} = {{{self.crit_values['negative']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}, df = {{{self.crit_values['degf']}}}$"))
            else:
                return ValueError("tails error in writing results")
            # determine significance
            if self.significance:
                print(f"reject the null hypothesis, results are significant, t({self.crit_values['degf']}) = {self.obt}, p < {self.alpha}")
            elif not self.significance:
                print(f"fail to reject the null hypothesis, results not significant, t({self.crit_values['degf']}) = {self.obt}, p > {self.alpha}")
            else:
                return ValueError("significance boolean error in writing results")
        elif self.test == "z":
            # print the critical value of t
            if self.tails == 2:
                display(Markdown(f"$z_{{crit}} = \\pm{{{self.crit_values['positive']}}}, \\alpha_{{two-tailed}} = {{{self.alpha}}}$"))
            elif self.tails == 1 and self.crit_values["direction"] == "increase":
                display(Markdown(f"$z_{{crit}} = +{{{self.crit_values['positive']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}$"))
            elif self.tails == 1 and self.crit_values["direction"] == "decrease":
                display(Markdown(f"$z_{{crit}} = {{{self.crit_values['negative']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}$"))
            else:
                return ValueError("tails error in writing results")
            # determine significance
            if self.significance:
                print(f"reject the null hypothesis, results are significant, z = {self.obt}, p < {self.alpha}, d = {self.effect_size}")
            elif not self.significance:
                print(f"fail to reject the null hypothesis, results not significant, z = {self.obt}, p > {self.alpha}, d = {self.effect_size}")
            else:
                return ValueError("significance boolean error in writing results")
        else:
            return ValueError("test specificaion error when writing results")
        
        


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


    def critical_value(self):
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
                crit = round(stats.norm.ppf(1 - self.alpha), 2)
            else:
                return ValueError("improper test specification - selecting crit values")
        elif self.tails == 2:
            if self.test in ["independent-samples t-test", "one-sample t-test", "dependent-samples t-test"]: 
                crit = round(stats.t.ppf(1 - self.alpha/2, degf), 2)
            elif self.test == "z":
                crit = round(stats.norm.ppf(1 - self.alpha/2), 2)
            else:
                return ValueError("improper test specification - selecting crit values")
        else:
            return ValueError("self.tails must equal 1 or 2")
        
        self.crit_values = {"positive": crit, "negative": -crit, "degf": degf}

        # add a direction for one-tailed tests
        if self.tails == 1:  
            direction = random.choice(["increase", "decrease"])
            self.crit_values["direction"] = direction
        else:
            return ValueError("tails must be 1 for directional crit values")
        
        return self.crit_values


    def z_test(self):
        if len(self.df.columns) > 1:
            raise Exception("Data contains more than one sample")
        elif len(self.df.columns) == 0:
            raise Exception("Dataframe error: no data columns")
        else:
            self.test = "z"

            # set the null and write out the question
            self.set_null_hypothesis()
            self.generate_question()

            # calculate the standard error
            # TODO double check the work here to make sure it is accurate
            sd = round(self.df['0'].std(ddof = 0), 2)
            n = len(self.df['0'])
            sem = round(sd/(round(math.sqrt(n),2)),2)
            self.obt = round((self.means[0] - self.null) / sem, 2)
            self.effect_size = round((self.means[0] - self.null) / sd, 2)

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
            display(Markdown(f"$z_{{obt}} = {{{self.obt}}}$"))
            print() # blank space
            # print calculations for cohen's d
            display(Markdown("calculating Cohen's d..."))
            display(Markdown("Cohen's d = $\\frac{{M - \\mu}}{{\\sigma}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{{self.means[0]} - {self.null}}}{{{sd}}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{{self.means[0] - self.null}}}{{{sd}}}$"))
            display(Markdown(f"Cohen's d = ${{{self.effect_size}}}$"))
            print() # blank space

            self.critical_value()
            self.significance = self.final_decision()
            self.write_result()

            # return self.obt - not returning b/c it was printing out the value of self.obt.  need to figure out why but commenting out fixed it


    def one_sample_t_test(self):
        if len(self.df.columns) > 1:
            raise Exception("Data contains more than one sample")
        elif len(self.df.columns) == 0:
            raise Exception("Dataframe error: no data columns")
        else:
            self.test = "one-sample t-test"   
            
            # set the null and write out the question
            self.set_null_hypothesis()
            self.generate_question()

            # calculate the standard error
            sem = round(math.sqrt(round((self.var[0]/self.n),2)),2)
            self.obt = round((self.means[0] - self.null) / sem, 2)
            self.effect_size = round((self.means[0] - self.null) / self.std[0], 2)

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
            display(Markdown(f"$t_{{obt}} = {{{self.obt}}}$"))
            print() # blank space
            # print calculations for cohen's d
            display(Markdown("calculating Cohen's d..."))
            display(Markdown("Cohen's d = $\\frac{{M - \\mu}}{{s}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{{self.means[0]} - {self.null}}}{{{self.std[0]}}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{{self.means[0] - self.null}}}{{{self.std[0]}}}$"))
            display(Markdown(f"Cohen's d = ${{{self.effect_size}}}$"))
            print() # blank space

            self.critical_value()
            self.significance = self.final_decision()
            self.write_result()

            # return self.obt - not returning b/c it was printing out the value of self.obt.  need to figure out why but commenting out fixed it        


    def independent_samples_t_test(self):
        if len(self.df.columns) == 1 or len(self.df.columns) > 2:
            raise ValueError("Data does not contain two samples")
        elif len(self.df.columns) == 0:
            raise ValueError("Dataframe error: no data columns")
        else: 
            self.test = "independent-samples t-test"           
                        
            # set the null and write out the question
            self.set_null_hypothesis()
            self.generate_question()
            
            # primary calculations
            pooled_var = round(((self.ss[0] + self.ss[1]) / ((self.n - 1) + (self.n - 1))), 2)
            sem = round(math.sqrt((round((pooled_var/self.n),2))+(round((pooled_var/self.n),2))),2)
            self.obt = round(((self.means[0] - self.means[1]) - self.null) / sem, 2)
            self.effect_size = round(((self.means[0] - self.means[1])) / round(math.sqrt(pooled_var),2), 2)

            # TODO adapt to display in the terminal or a notebook
            # display the caluclations for the pooled variance
            print("calculating the pooled variance...")
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
            display(Markdown(f"$t_{{obt}} = {{{self.obt}}}$"))
            print() # blank space
            # print calculations for cohen's d
            display(Markdown("calculating Cohen's d..."))
            display(Markdown("Cohen's d = $\\frac{{M)0 - M_1}}{{\\sqrt{{{s_p^2}}}}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{({self.means[0]} - {self.means[1]})}}{{{{{{\\sqrt{{{pooled_var}}}}}}}}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{({self.means[0] - self.means[1]})}}{{{round(math.sqrt(pooled_var),2)}}}$"))
            display(Markdown(f"Cohen's d = ${{{self.effect_size}}}$"))
            print() # blank space

            self.critical_value()
            self.significance = self.final_decision()
            self.write_result()

            # return self.obt - not returning b/c it was printing out the value of self.obt.  need to figure out why but commenting out fixed it
        

    def dependent_samples_t_test(self):
        if len(self.df.columns) == 1 or len(self.df.columns) > 2:
            raise ValueError("Data does not contain two samples")
        elif len(self.df.columns) == 0:
            raise ValueError("Dataframe error: no data columns")
        else:
            self.test = "dependent-samples t-test"     
            
            # set the null and write out the question
            self.set_null_hypothesis()
            self.generate_question()

            # primary calculations
            # need to gather the difference scores
            self.df['D'] = self.df['1'] - self.df['0']
            
            # print the dataframe with the difference scores
            display(Markdown("Calculating the difference scores $D = X_1 - X_0$"))
            print(self.df.to_string(index = False))
            print() # blank space
            # Calculate the Mean of the Difference Scores
            sum_d = self.df['D'].sum()
            n = len(self.df['D'])
            mean_d = round(sum_d/n, 2)
            display(Markdown("Calculating the Mean of the Difference Scores..."))
            display(Markdown("$M_D = \\frac{{\\Sigma D}}{{n}}$"))
            display(Markdown(f"$M_D = \\frac{{{sum_d}}}{{{n}}}$"))
            display(Markdown(f"$M_D = {{{mean_d}}}$"))
            print() # blank space
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
            print() # blank space
            # calculate the variance    
            variance = round(ss / (n - 1), 2)
            display(Markdown("$ s^2 = \\frac{{SS_D}}{{df}}$")) 
            display(Markdown(f"$ s^2 = \\frac{{{ss}}}{{{n - 1}}}$"))   
            display(Markdown(f"$ s^2 = \\frac{{{round(ss/(n - 1), 2)}}}$"))
            display(Markdown(f"$ s^2 = {{{variance}}}$"))  
            print() # blank space
            # Calculate the estimated standard error
            sem = round(math.sqrt(variance/n), 2)
            display(Markdown("Calculating the estimated standard error..."))
            display(Markdown("$ s_{M_D} = \\sqrt{{\\frac{{s^2}}{{n}}}}$"))
            display(Markdown(f"$ s_{{M_D}} = \\sqrt{{\\frac{{{variance}}}{{{n}}}}}$"))
            display(Markdown(f"$ s_{{M_D}} = \\sqrt{{{round(variance/n, 2)}}}$"))
            display(Markdown(f"$ s_{{M_D}} = {{{sem}}}$"))
            print() # blank space
            # caclulate the t-statistic
            self.obt = round((mean_d - self.null) / sem, 2)
            display(Markdown("calculating $t_{{obt}}$..."))
            display(Markdown("$t_{{obt}} = {{\\frac{{M_D - \\mu_D}}{{s_{M_D}}}}}$"))
            display(Markdown(f"$t_{{obt}} = \\frac{{{mean_d} - {self.null}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = \\frac{{{mean_d - self.null}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = {{{self.obt}}}$"))
            print() # blank space
            # print calculations for cohen's d
            self.effect_size = self.obt = round(mean_d / round(math.sqrt(variance),2), 2)
            display(Markdown("calculating Cohen's d..."))
            display(Markdown("Cohen's d = $\\frac{{M_D}}{{\\sqrt{{s^2}}}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{{mean_d}}}{{{{{{\\sqrt{{{variance}}}}}}}}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{{mean_d}}}{{{round(math.sqrt(variance),2)}}}$"))
            display(Markdown(f"Cohen's d = ${{{self.effect_size}}}$"))
            print() # blank space

            self.critical_value()
            self.significance = self.final_decision()
            self.write_result()

            # return self.obt - not returning b/c it was printing out the value of self.obt.  need to figure out why but commenting out fixed it