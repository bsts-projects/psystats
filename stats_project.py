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
        self.sums = self.col_sums()
        self.g = self.anova_g()
        self.sum_squared_scores = self.grand_sum_squared_scores()
        self.var = self.variance()
        self.std = self.stdev()
        self.test = "" # value is set when a stats function is called
        self.alpha = self.set_alpha()
        self.tails = int
        self.null = int
        self.obt = float
        self.effect_size = float
        self.crit_values = {} # dict (it was originally typed {}; dict but I have no idea why)
        self.significance = bool
        if distribution == "normal":
            self.distribution = distribution 
        else:
            raise ValueError("only the normal distribution is currently supported")
            # TODO add the ability to generate data from other distribuions


    def set_alpha(self):
        self.alpha = random.choice([0.05, 0.01])
        return self.alpha
        
    
    def generate_data(self):
        df = pd.DataFrame()
        # list of letters for group labels
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # create data for each group and add it to the dataframe
        for group in range(self.groups):
            mean = random.randint(10, 50)
            sd = mean * random.uniform(0.10, 0.50)
            self.pop_sd = round(sd) # creating a variable to log the pop SD
            self.pop_mean = mean # variable to log the pop mean
            # generate the sample based on the above values

            same_diff = random.randint(0,3)
            if same_diff >= 1:
                effect =  round(mean * random.uniform(0.10, 0.40))
                mean +=  effect
            samples = np.random.normal(mean, sd, self.n)

            # round the data so it only includes whole numbers
            sample = np.round(samples).astype(int)

            # convert to a dataframe to display the data
            df[f'{letters[group]}'] = sample
        return df
    

    def generate_question(self):
        # determine the test type

        if self.tails == 2:
            text = "significantly different from"
        elif self.tails == 1:
            if self.crit_values["direction"] == "increase" :
                text = "significantly greater than"
            elif self.crit_values["direction"] == "decrease":
                text = "significantly less than"
            else:
                return ValueError("direction error for generating question")
        else:
            return ValueError("tails error for question generation")    

        #display(self.df.style.hide(axis="index"))
        
        if self.test == "z":
            display(Markdown(f"Given the following data, is the mean of $Group_A$ {text} the population mean: $\\mu = {{{self.null}}}$?<br><br>Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$<br><br>"))
            display(self.df.style.hide(axis="index"))
            display(Markdown(f"""<br>The necessary summary statistics for these data<br>
                            $$M_A = {{{self.means[0]}}}$$
                            $${{\\sigma}} = {{{self.pop_sd}}}$$
                            $$n = {{{len(self.df['A'])}}}$$<br><br>
                            """)) #${{\\sigma_A}} = {{{round(self.df['A'].std(ddof = 0), 2)}}}$<br><br>
        elif self.test == "one-sample t-test":
            display(Markdown(f"Given the following data, is the mean of $Group_A$ {text} ${{{self.null}}}$?  Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$ <br><br>"))
            #display(Markdown(self.df.to_markdown(index=False)))
            display(self.df.style.hide(axis="index"))
            display(Markdown(f"""<br><br>
                                $M_A = {{{self.means[0]}}}$<br><br>
                                $s^2 = {{{self.var[0]}}}$<br><br>
                                $n = {{{len(self.df['A'])}}}$<br><br>
                                """))
        elif self.test == "independent-samples t-test":
            display(Markdown(f"Given the following between-subjects data, is the mean of $Group_A$ {text} the mean of $Group_B$?  Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$"))
            display(Markdown(f"$M_A = {{{self.means[0]}}}, M_B = {{{self.means[1]}}}$"))
            display(Markdown(f"$SS_A = {{{self.ss[0]}}}, SS_B = {{{self.ss[1]}}}$"))
            display(Markdown(f"$ n_A = {{{len(self.df['A'])}}}, n_B = {{{len(self.df['B'])}}}$"))
        elif self.test == "dependent-samples t-test":
            display(Markdown(f"Given the following within-subjects data, is $M_D$ {text} ${{{self.null}}}$?  Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$"))
            display(Markdown(f"$M_A = {{{self.means[0]}}}, M_B = {{{self.means[1]}}}$"))
            display(Markdown(f"$ n = {{{len(self.df['A'])}}}$"))
        elif self.test == "one-way ANOVA":
            display(Markdown(f"Given the following between-subjects data, use a one-way ANOVA with $\\alpha = {{{self.alpha}}}$"))
            display(Markdown(f"$G = {{{self.g}}}, \\Sigma X^2 = {{{self.sum_squared_scores}}}, k = {{{self.groups}}}, N = {{{self.groups * self.n}}}$"))
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for group in range(self.groups):
                display(Markdown(f"$T_{{{letters[group]}}} = {{{self.sums[group]}}}, SS_{{{letters[group]}}} = {{{self.ss[group]}}}$"))   
        elif self.test == "repeated-measures ANOVA":
            display(Markdown(f"Given the following within-subjects data, use a repeated-measures ANOVA with $\\alpha = {{{self.alpha}}}$"))
            display(Markdown(f"$G = {{{self.g}}}, \\Sigma X^2 = {{{self.sum_squared_scores}}}, k = {{{self.groups}}}, N = {{{self.groups * self.n}}}$"))
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for group in range(self.groups):
                display(Markdown(f"$T_{{{letters[group]}}} = {{{self.sums[group]}}}, SS_{{{letters[group]}}} = {{{self.ss[group]}}}$"))  
        else:
            return ValueError("test-type specification error in question geneneration")
       
        
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
                display(Markdown(f"$t_{{crit}} = \\pm{{{self.crit_values['positive']}}}, \\alpha_{{two-tailed}} = {{{self.alpha}}}, df = {{{self.crit_values['degf']}}}$<br><br>"))
            elif self.tails == 1 and self.crit_values["direction"] == "increase":
                display(Markdown(f"$t_{{crit}} = +{{{self.crit_values['positive']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}, df = {{{self.crit_values['degf']}}}$<br><br>"))
            elif self.tails == 1 and self.crit_values["direction"] == "decrease":
                display(Markdown(f"$t_{{crit}} = {{{self.crit_values['negative']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}, df = {{{self.crit_values['degf']}}}$<br><br>"))
            else:
                return ValueError("tails error in writing results")
            # determine significance
            if self.significance:
                print(f"reject the null hypothesis, results are significant, t({self.crit_values['degf']}) = {self.obt}, p < {self.alpha}, d = {self.effect_size}<br><br>")
            elif not self.significance:
                print(f"fail to reject the null hypothesis, results not significant, t({self.crit_values['degf']}) = {self.obt}, p > {self.alpha}, d = {self.effect_size}<br><br>")
            else:
                return ValueError("significance boolean error in writing results")
        elif self.test == "z":
            # print the critical value of t
            display(Markdown("The decision criteria: <br><br>"))
            if self.tails == 2:
                display(Markdown(f"$z_{{crit}} = \\pm{{{self.crit_values['positive']}}}, \\alpha_{{two-tailed}} = {{{self.alpha}}}$<br><br>"))
            elif self.tails == 1 and self.crit_values["direction"] == "increase":
                display(Markdown(f"$z_{{crit}} = +{{{self.crit_values['positive']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}$<br><br>"))
            elif self.tails == 1 and self.crit_values["direction"] == "decrease":
                display(Markdown(f"$z_{{crit}} = {{{self.crit_values['negative']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}$<br><br>"))
            else:
                return ValueError("tails error in writing results")
            # determine significance
            display(Markdown("The results: <br><br>"))
            if self.significance:
                display(Markdown(f"reject the null hypothesis, results are significant, *z* = {self.obt}, *p* < {self.alpha}, *d* = {self.effect_size}<br><br>"))
            elif not self.significance:
                display(Markdown(f"fail to reject the null hypothesis, results not significant, *z* = {self.obt}, *p* > {self.alpha}, *d* = {self.effect_size}<br><br>"))
            else:
                return ValueError("significance boolean error in writing results")
        elif self.test in ["one-way ANOVA", "repeated-measures ANOVA"]:
            display(Markdown(f"$F_{{crit}} = {{{self.crit_values['positive']}}}, \\alpha = {{{self.alpha}}}$"))
            if self.significance:
                display(Markdown(f"reject the null hypothesis, results are significant, $ F({{{self.crit_values['degf_n']}}}, {{{self.crit_values['degf_d']}}}) = {{{self.obt}}}, p < {{{self.alpha}}}, \\eta^2 = {{{self.effect_size}}}$"))
            elif not self.significance:
                display(Markdown(f"fail to reject the null hypothesis, results not significant, $ F({{{self.crit_values['degf_n']}}}, {{{self.crit_values['degf_d']}}}) = {{{self.obt}}}, p > {{{self.alpha}}}, \\eta^2 = {{{self.effect_size}}}$"))

        else:
            return ValueError("test specificaion error when writing results")
        
        
    def set_null_hypothesis(self):
        # for one sample tests, sets a null hypothess between -2 to + 2 x the mean
        if  self.test in ["one-sample t-test"]:
            mean = self.means[0]
            multiplier = random.uniform(-2, 2)
            self.null = round(mean * multiplier)
        elif self.test == "z":
            self.null = self.pop_mean
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
    

    def col_sums(self):
        sums = []
        for column in self.df:
            sums.append(self.df[column].sum())
        return sums


    def anova_g(self):
        g = 0
        for column in self.df:
            g += self.df[column].sum()
        return g
    

    def grand_sum_squared_scores(self): # for anova
        sum_squared_scores = 0
        for column in self.df: 
            score = (self.df[column].apply(lambda x: x ** 2)).sum()  
            sum_squared_scores += score
        return sum_squared_scores
    

    def critical_value(self):
        # calculate the degrees of freedom based on the type of test used
        if self.test == "independent-samples t-test":
            degf = (self.n - 1) + (self.n - 1)
            self.tails = random.choice([1, 2])
            if self.tails == 1:
                crit = round(stats.t.ppf(1 - self.alpha, degf), 2) # type: ignore
            elif self.tails == 2:
                crit = round(stats.t.ppf(1 - self.alpha/2, degf), 2) # type: ignore
            else: 
                return ValueError("crit not determined")
            self.crit_values = {"positive": crit, "negative": -crit, "degf": degf}

        elif self.test == "one-sample t-test" or self.test == "dependent-samples t-test":
            degf = self.n - 1
            self.tails = random.choice([1, 2])
            if self.tails == 1:
                crit = round(stats.t.ppf(1 - self.alpha, degf), 2) # type: ignore
            elif self.tails == 2:
                crit = round(stats.t.ppf(1 - self.alpha/2, degf), 2) # type: ignore
            else: 
                return ValueError("crit not determined")
            self.crit_values = {"positive": crit, "negative": -crit, "degf": degf}

        # TODO manually specify crit values for z scores.  or figure out why the math is incorrect
        elif self.test == "z":
            self.tails = random.choice([1, 2])
            if self.tails == 1:
                crit = round(stats.norm.ppf(1 - self.alpha), 3) # type: ignore
            elif self.tails == 2:
                crit = round(stats.norm.ppf(1 - self.alpha/2), 2) # type: ignore
            else: 
                return ValueError("crit not determined")
            self.crit_values = {"positive": crit, "negative": -crit}
        
        elif self.test == "one-way ANOVA":
            degf_w = (self.n * self.groups) - self.groups
            degf_b = self.groups - 1
            self.tails = 1
            crit = round(stats.f.ppf(q = (1 - self.alpha), dfn = degf_b, dfd = degf_w), 2) # type: ignore
            self.crit_values = {"positive": crit, "degf_d": degf_w, "degf_n": degf_b}
            
        elif self.test == "repeated-measures ANOVA":
            degf_e = ((self.n * self.groups) - self.groups) - (self.n - 1)
            degf_b = self.groups - 1
            self.tails = 1
            crit = round(stats.f.ppf(q = (1 - self.alpha), dfn = degf_b, dfd = degf_e), 2) # type: ignore
            self.crit_values = {"positive": crit, "degf_d": degf_e, "degf_n": degf_b}


        else:
            raise ValueError("incorrect test specification - degrees of freedom")

        # add a direction for one-tailed tests 
        if self.test in ["one-way ANOVA", "repeated-measures ANOVA"]:
            self.crit_values["direction"] = "increase"
        else:
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
            self.critical_value()
            self.generate_question()

            # calculate the standard error
            # TODO double check the work here to make sure it is accurate
            sd = self.pop_sd # round(self.df['A'].std(ddof = 0), 2)
            n = len(self.df['A'])
            sem = round(sd/(round(math.sqrt(n),2)),2)
            self.obt = round((self.means[0] - self.null) / sem, 2)
            self.effect_size = round((self.means[0] - self.null) / sd, 2)

            # TODO add a way to determine environment so output can display in terminal or notebook
            # New output formatted for quarto render to html and screen reader
            display(Markdown(f"""Calculate the standard error <br>
                            $$\\sigma_M = \\frac{{\\sigma}}{{\\sqrt{{N}}}}$$ <br>
                            $$\\sigma_M = \\frac{{{sd}}}{{\\sqrt{{{n}}}}}$$ <br>
                            $$\\sigma_M = \\frac{{{sd}}}{{{round(math.sqrt(n),2)}}}$$ <br>
                            $$\\sigma_M = {{{sem}}}$$ <br><br>
                            Calculate $z_{{obt}}$ <br>
                            $$z_{{obt}} = {{\\frac{{M - \\mu}}{{\\sigma_M}}}}$$ <br>
                            $$z_{{obt}} = \\frac{{{self.means[0]} - {self.null}}}{{{sem}}}$$ <br>
                            $$z_{{obt}} = \\frac{{{round(self.means[0] - self.null, 2)}}}{{{sem}}}$$ <br>
                            $$z_{{obt}} = {{{self.obt}}}$$ <br><br>
                            Calculate Cohen's d for effect size <br>
                            $$d = \\frac{{M - \\mu}}{{\\sigma}}$$ <br>
                            $$d = \\frac{{{self.means[0]} - {self.null}}}{{{sd}}}$$ <br>
                            $$d = \\frac{{{round(self.means[0] - self.null, 2)}}}{{{sd}}}$$ <br>
                            $$d = {{{self.effect_size}}}$$ <br><br>"""))
            
            
            # old output version
            '''
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
            '''

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
            self.critical_value()
            self.generate_question()

            # calculate the standard error
            sem = round(math.sqrt(round((self.var[0]/self.n),2)),2)
            self.obt = round((self.means[0] - self.null) / sem, 2)
            self.effect_size = round((self.means[0] - self.null) / self.std[0], 2)

            # print the caluclations for the standard error
            # TODO add a way to determine environment so output can display in terminal or notebook
            display(Markdown(f"""calculating the standard error <br><br>
                            $s_M = \\sqrt{{\\frac{{s^2}}{{n}}}}$ <br><br>
                            $s_M = \\sqrt{{\\frac{{{self.var[0]}}}{{{self.n}}}}}$ <br><br>
                            $s_M = \\sqrt{{{round((self.var[0]/self.n),2)}}}$ <br><br>
                            $s_M = {{{sem}}}$ <br><br>
                            calculating $t_{{obt}}$ <br><br>
                            $t_{{obt}} = {{\\frac{{M - \\mu}}{{s_M}}}}$ <br><br>
                            $t_{{obt}} = \\frac{{{self.means[0]} - {self.null}}}{{{sem}}}$ <br><br>
                            $t_{{obt}} = \\frac{{{self.means[0] - self.null}}}{{{sem}}}$ <br><br>
                            $t_{{obt}} = {{{self.obt}}}$ <br><br>
                            calculating Cohen's d <br><br>
                            Cohen's $d = \\frac{{M - \\mu}}{{s}}$ <br><br>
                            Cohen's $d = \\frac{{{self.means[0]} - {self.null}}}{{{self.std[0]}}}$ <br><br>
                            Cohen's $d = \\frac{{{self.means[0] - self.null}}}{{{self.std[0]}}}$ <br><br>
                            Cohen's $d = {{{self.effect_size}}}$ <br><br>
                            """))

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
            self.critical_value()
            self.generate_question()
            
            # primary calculations
            pooled_var = round(((self.ss[0] + self.ss[1]) / ((self.n - 1) + (self.n - 1))), 2)
            sem = round(math.sqrt((round((pooled_var/self.n),2))+(round((pooled_var/self.n),2))),2)
            self.obt = round(((self.means[0] - self.means[1]) - self.null) / sem, 2)
            self.effect_size = round(((self.means[0] - self.means[1])) / round(math.sqrt(pooled_var),2), 2)

            # TODO adapt to display in the terminal or a notebook
            # display the caluclations for the pooled variance
            print("calculating the pooled variance...")
            display(Markdown("$s_p^2 = {{\\frac{{SS_A + SS_B}}{{df_A + df_B}}}}$"))
            display(Markdown(f"$s_p^2 = {{\\frac{{{self.ss[0]} + {self.ss[1]}}}{{{self.n - 1} + {self.n - 1}}}}}$"))
            display(Markdown(f"$s_p^2 = \\frac{{{round(self.ss[0] + self.ss[1],2)}}}{{{(self.n - 1) + (self.n - 1)}}}$"))
            display(Markdown(f"$s_p^2 = {{{pooled_var}}}$"))
            # display the calculations for the estimated standard error
            print("calculating the estimated standard error of the difference between means...")
            display(Markdown("$s_{{(M_A - M_B)}} = \\sqrt{{\\frac{{s_p^2}}{{n_1}} + \\frac{{s_p^2}}{{n_1}}}}$"))
            display(Markdown(f"$s_{{(M_A - M_B)}} = \\sqrt{{\\frac{{{pooled_var}}}{{{self.n}}} + \\frac{{{pooled_var}}}{{{self.n}}}}}$"))
            display(Markdown(f"$s_{{(M_A - M_B)}} = \\sqrt{{{round(pooled_var/self.n, 2)} + {round(pooled_var/self.n, 2)}}}$"))
            display(Markdown(f"$s_{{(M_A - M_B)}} = \\sqrt{{{round(pooled_var/self.n, 2) + round(pooled_var/self.n, 2)}}}$"))
            display(Markdown(f"$s_{{(M_A - M_B)}} = {{{sem}}}$"))
            # display the caluclations for t_obt
            display(Markdown("calculating $t_{{obt}}$..."))
            display(Markdown(f"$t_{{obt}} = {{\\frac{{(M_A - M_B) - (\\mu_A - \\mu_B)}}{{s_{{(M_A - M_B)}}}}}}$"))
            display(Markdown(f"$t_{{obt}} = \\frac{{({self.means[0]} - {self.means[1]}) - {{{self.null}}}}}{{{sem}}}$")) 
            display(Markdown(f"$t_{{obt}} = \\frac{{{round(self.means[0] - self.means[1] - self.null, 2)}}}{{{sem}}}$"))
            display(Markdown(f"$t_{{obt}} = {{{self.obt}}}$"))
            print() # blank space
            # print calculations for cohen's d
            display(Markdown("calculating Cohen's d..."))
            display(Markdown("Cohen's d = $\\frac{{M_A - M_B}}{{\\sqrt{{{s_p^2}}}}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{({self.means[0]} - {self.means[1]})}}{{{{{{\\sqrt{{{pooled_var}}}}}}}}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{({self.means[0] - self.means[1]})}}{{{round(math.sqrt(pooled_var),2)}}}$"))
            display(Markdown(f"Cohen's d = ${{{self.effect_size}}}$"))
            print() # blank space

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
            self.critical_value()
            self.generate_question()

            # primary calculations
            # need to gather the difference scores
            self.df['D'] = self.df['B'] - self.df['A']
            
            # print the dataframe with the difference scores
            display(Markdown("Calculating the difference scores $D = X_B - X_A$"))
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
            self.effect_size = round(mean_d / round(math.sqrt(variance),2), 2)
            display(Markdown("calculating Cohen's d..."))
            display(Markdown("Cohen's d = $\\frac{{M_D}}{{\\sqrt{{s^2}}}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{{mean_d}}}{{{{{{\\sqrt{{{variance}}}}}}}}}$"))
            display(Markdown(f"Cohen's d = $\\frac{{{mean_d}}}{{{round(math.sqrt(variance),2)}}}$"))
            display(Markdown(f"Cohen's d = ${{{self.effect_size}}}$"))
            print() # blank space

            self.significance = self.final_decision()
            self.write_result()

            # return self.obt - not returning b/c it was printing out the value of self.obt.  need to figure out why but commenting out fixed it


    def anova(self, test: str):
        if len(self.df.columns) == 1:
            raise ValueError("Data does not contain at least two samples")
        elif len(self.df.columns) == 0:
            raise ValueError("Dataframe error: no data columns")
        else:
            if test == "one-way":
                self.test = "one-way ANOVA"     
            elif test == "repeated-measures":
                self.test = "repeated-measures ANOVA"
            else:
                raise ValueError("test must be set to: 'one-way', 'repeated-measures'")

            # set the null and write out the question
            self.set_null_hypothesis()
            self.critical_value()
            self.generate_question()

            # Primary Calculations
            big_n = self.groups * self.n
            
            # degrees of freedom
            df_total = big_n - 1
            df_between = self.groups - 1
            df_within = big_n - self.groups

            print() # blank space
            if test == "repeated-measures":
                print("Stage 1 Calculations:")

            print("calculating the degrees of freedom...")
    
            display(Markdown(f"$df_{{total}} = N - 1$"))
            display(Markdown(f"$df_{{total}} = {{{big_n}}} - 1$"))
            display(Markdown(f"$df_{{total}} = {{{df_total}}}$"))
            print() # blank space

            display(Markdown(f"$df_{{between}} = k - 1 $"))
            display(Markdown(f"$df_{{between}} = {{{self.groups}}} - 1 $"))
            display(Markdown(f"$df_{{between}} = {{{df_between}}}$"))
            print() # blank space

            display(Markdown(f"$df_{{within}} = N - K $"))
            display(Markdown(f"$df_{{within}} = {{{big_n}}} - {{{self.groups}}} $"))
            display(Markdown(f"$df_{{within}} = {{{df_within}}}$"))
            print() # blank space

            # sum of squares
            ss_total = self.sum_squared_scores - round(((self.g**2)/big_n), 2)
            ss_within = 0
            for group in range(self.groups):
                ss_within += self.ss[group]
            ss_between = ss_total - ss_within

            print("calulating the sum of squares...")
            display(Markdown(f"$ SS_{{total}} = \\Sigma X^2 - \\frac{{G^2}}{{N}} $"))
            display(Markdown(f"$ SS_{{total}} = {{{self.sum_squared_scores}}} - \\frac{{{self.g}^2}}{{{big_n}}} $"))
            display(Markdown(f"$ SS_{{total}} = {{{self.sum_squared_scores}}} - \\frac{{{self.g**2}}}{{{big_n}}} $"))
            display(Markdown(f"$ SS_{{total}} = {{{self.sum_squared_scores}}} - {{{round(((self.g**2)/big_n), 2)}}} $"))
            display(Markdown(f"$ SS_{{total}} = {{{round(ss_total, 2)}}} $"))
            print() # blank space

            display(Markdown(f"$ SS_{{within}} = \\Sigma SS_{{inside\\_each\\_condition}} $"))
            values = ""
            for group in range(self.groups):
                if group == 0:
                    values += f"{self.ss[group]}"
                else:
                    values += f" + {self.ss[group]}"
            display(Markdown(f"$ SS_{{within}} = {{{values}}}$"))
            display(Markdown(f"$ SS_{{within}} = {{{round(ss_within, 2)}}}$"))
            print() # blank space

            display(Markdown(f"$ SS_{{between}} = SS_{{total}} - SS_{{within}} $"))
            display(Markdown(f"$ SS_{{between}} = {{{round(ss_total, 2)}}} - {{{round(ss_within, 2)}}} $"))
            display(Markdown(f"$ SS_{{between}} = {{{round(ss_between, 2)}}} $"))
            print() # blank space

            display(Markdown("note: the other way to calculate $SS_{{betwen}}$ is:"))
            display(Markdown("$ SS_{{between}} = \\Sigma{{\\frac{{T^2}}{{n}}}} - \\frac{{G^2}}{{N}} $"))
            print() # blank space

            if self.test == "repeated-measures ANOVA":
                # degrees of freedom
                df_subjects = self.n - 1
                df_error = df_within - df_subjects
                
                print("Stage 2 Calculations:")
                print("partitioning the degrees of freedom...")

                display(Markdown(f"$df_{{subjects}} = n - 1 $"))
                display(Markdown(f"$df_{{subjects}} = {{{self.n}}} - 1 $"))
                display(Markdown(f"$df_{{subjects}} = {{{df_subjects}}}$"))
                print() # blank space

                display(Markdown(f"$df_{{error}} = df_{{within}} - df_{{subjects}} $"))
                display(Markdown(f"$df_{{error}} = {{{df_within}}} - {{{df_subjects}}} $"))
                display(Markdown(f"$df_{{error}} = {{{df_error}}} $"))
                print() # blank space

                # sum of squares
                print("partitioning the sum of squares...")
                
                self.df["P"] = self.df.sum(axis = 1)
                print(self.df)
                print() # blank space

                p_sums = [] # holder for the P sums for displaying below
                p_squared = [] # holds the squared partipant sums (P^2)
                quotients = [] # holds the value of each (P^2)/k
                sum_quotients = 0 # holds the sum of (P^2)/k
                for p in self.df["P"]:
                    p_sums.append(p)
                    p_squared.append(p ** 2)
                    quotients.append(round((p **2 / self.groups), 2))
                    sum_quotients += round((p **2 / self.groups), 2)

                ss_subjects = round(sum_quotients, 2) - round(((self.g**2)/big_n), 2)
                ss_error = ss_within - ss_subjects

                display(Markdown(f"$SS_{{subjects}} = \\Sigma{{\\frac{{P^2}}{{k}}}} - \\frac{{G^2}}{{N}} $"))
                temp_text = ""
                for p in p_sums:
                    temp_text += f" + \\frac{{{p}^2}}{{{self.groups}}}"
                display(Markdown(f"$SS_{{subjects}} = {{{temp_text[3:]}}} - \\frac{{{self.g}^2}}{{{big_n}}}$"))
                temp_text = ""
                for p in p_squared:
                    temp_text += f" + \\frac{{{p}}}{{{self.groups}}}"
                display(Markdown(f"$SS_{{subjects}} = {{{temp_text[3:]}}} - \\frac{{{self.g**2}}}{{{big_n}}}$"))
                temp_text = ""
                for p in quotients:
                    temp_text += f" + {{{p}}}"
                display(Markdown(f"$SS_{{subjects}} = {{{temp_text[3:]}}} - {{{round(((self.g**2)/big_n), 2)}}}$"))
                display(Markdown(f"$SS_{{subjects}} = {{{round(sum_quotients,2)}}} - {{{round(((self.g**2)/big_n), 2)}}}$")) 
                display(Markdown(f"$SS_{{subjects}} = {{{round(ss_subjects,2)}}}$"))     
                print() # blank space

                display(Markdown(f"$SS_{{error}} = SS_{{within}} - SS_{{subjects}} $"))
                display(Markdown(f"$SS_{{error}} = {{{round(ss_within,2)}}} - {{{round(ss_subjects,2)}}}$"))
                display(Markdown(f"$SS_{{error}} = {{{round(ss_error,2)}}}$"))
                print() # blank space


            # mean squares
            ms_between = round(round(ss_between, 2)/df_between, 2)
            ms_within = round(round(ss_within, 2)/df_within, 2)
            

            print("calculating the mean squares...")
            display(Markdown(f"$ MS_{{between}} = \\frac{{SS_{{between}}}}{{df_{{between}}}} $"))
            display(Markdown(f"$ MS_{{between}} = \\frac{{{round(ss_between, 2)}}}{{{df_between}}} $"))
            display(Markdown(f"$ MS_{{between}} = {{{round(ms_between, 2)}}} $"))
            print() # blank space

            if self.test == "one-way ANOVA":
                display(Markdown(f"$ MS_{{within}} = \\frac{{SS_{{within}}}}{{df_{{within}}}} $"))
                display(Markdown(f"$ MS_{{within}} = \\frac{{{round(ss_within, 2)}}}{{{df_within}}} $"))
                display(Markdown(f"$ MS_{{within}} = {{{round(ms_within, 2)}}} $"))
                print() # blank space
                
                # F value
                self.obt = round(ms_between/ms_within, 2)

                print("calculating the f ratio...")
                display(Markdown(f"$ F_{{obt}} = \\frac{{MS_{{between}}}}{{MS_{{within}}}} $"))
                display(Markdown(f"$ F_{{obt}} = \\frac{{{round(ms_between,2)}}}{{{round(ms_within,2)}}} $"))
                display(Markdown(f"$ F_{{obt}} = {{{round(self.obt, 2)}}} $"))
                print() # blank space

                # effect size
                self.effect_size = round(round(ss_between, 2)/round(ss_total, 2), 2)
                
                print("calculating eta squared...")
                display(Markdown(f"$ \\eta^2 = \\frac{{SS{{between}}}}{{SS_{{total}}}} $"))
                display(Markdown(f"$ \\eta^2 = \\frac{{{round(ss_between, 2)}}}{{{round(ss_total, 2)}}} $"))
                display(Markdown(f"$ \\eta^2 = {{{round(self.effect_size, 2)}}} $"))
                print() # blank space


            elif test == "repeated-measures":
                ms_error = round(round(ss_error,2)/df_error, 2) # type: ignore

                display(Markdown(f"$ MS_{{error}} = \\frac{{SS_{{error}}}}{{df_{{error}}}} $"))
                display(Markdown(f"$ MS_{{error}} = \\frac{{{round(ss_error, 2)}}}{{{df_error}}} $")) # type: ignore
                display(Markdown(f"$ MS_{{error}} = {{{round(ms_error, 2)}}} $"))
                print() # blank space

                # F value
                self.obt = round(ms_between/ms_error, 2)

                print("calculating the f ratio...")
                display(Markdown(f"$ F_{{obt}} = \\frac{{MS_{{between}}}}{{MS_{{error}}}} $"))
                display(Markdown(f"$ F_{{obt}} = \\frac{{{round(ms_between,2)}}}{{{round(ms_error,2)}}} $"))
                display(Markdown(f"$ F_{{obt}} = {{{round(self.obt, 2)}}} $"))
                print() # blank space

                # effect size
                self.effect_size = round(round(ss_between, 2)/(round(ss_total, 2) - round(ss_subjects, 2)), 2) # type: ignore
                
                print("calculating partial eta squared...")
                display(Markdown(f"$ \\eta_p^2 = \\frac{{SS{{between}}}}{{SS_{{total}} - SS_{{subjects}}}} $"))
                display(Markdown(f"$ \\eta_p^2 = \\frac{{{round(ss_between, 2)}}}{{{{{round(ss_total, 2)}}} - {{{round(ss_subjects, 2)}}}}} $")) # type: ignore
                display(Markdown(f"$ \\eta_p^2 = \\frac{{{round(ss_between, 2)}}}{{{round(ss_total, 2) - round(ss_subjects, 2)}}} $")) # type: ignore
                display(Markdown(f"$ \\eta_p^2 = {{{round(self.effect_size, 2)}}} $"))
                print() # blank space

            else:
                raise ValueError("test accepts: 'one-way', 'repeated-measures'")

            self.significance = self.final_decision()
            self.write_result()
            print() # blank space