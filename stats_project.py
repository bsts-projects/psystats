import numpy as np
import pandas as pd
import random, math
import itertools
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
        
        self.pop_mean = random.randint(10, 50)
        self.pop_sd = round(self.pop_mean * random.uniform(0.10, 0.30))
            
        # create data for each group and add it to the dataframe
        for group in range(self.groups):
            mean = self.pop_mean #random.randint(10, 50)
            sd = self.pop_sd
            # generate the sample based on the above values

            same_diff = random.randint(0,3)
            if same_diff >= 1:
                effect =  round(mean * random.uniform(0.10, 0.50))
                mean +=  effect
            samples = np.random.normal(mean, sd, self.n)

            # round the data so it only includes whole numbers
            sample = np.round(samples).astype(int)

            # convert to a dataframe to display the data
            df[f'{letters[group]}'] = sample
        return df
    

    def set_test(self, test: str):
        # method to to be called from within the factorial class to set value
        self.test = test
        self.tails = 1
        self.crit_values["direction"] = "increase"


    def set_crit_values(self, value, degf_n: int, degf_d: int):
        # method to to be called from within the factorial class to set value
        self.crit_values.update({"positive": value, "degf_d": degf_d, "degf_n": degf_n})


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

        # add ID column for repeated measures data
        if self.test in ["dependent-samples t-test", "repeated-measures ANOVA"]:
            id_col = list(range(1, self.n + 1))
            self.df.insert(loc = 0, column = "ID", value = id_col)
        
        if self.test == "z":
            display(Markdown(f"Given the following data, is the mean of $Group_A$ {text} the population mean: $\\mu = {{{self.null}}}$? <br><br>Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$<br><br>"))
            display(self.df.style.hide(axis="index"))
            display(Markdown(f"""<br>The necessary summary statistics for these data<br>
                            $$M_A = {{{self.means[0]}}}$$
                            $${{\\sigma}} = {{{self.pop_sd}}}$$
                            $$n = {{{len(self.df['A'])}}}$$<br>
                            """)) 
        elif self.test == "one-sample t-test":
            display(Markdown(f"Given the following data, is the mean of $Group_A$ {text} ${{{self.null}}}$?  Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$ <br><br>"))
            display(self.df.style.hide(axis="index"))
            display(Markdown(f"""<br> The necessary summary statistics for these data<br>
                                $$M_A = {{{self.means[0]}}}$$
                                $$s^2 = {{{self.var[0]}}}$$
                                $$n = {{{len(self.df['A'])}}}$$<br>
                                """))
        elif self.test == "independent-samples t-test":
            display(Markdown(f"Given the following between-subjects data, is the mean of $Group_A$ {text} the mean of $Group_B$?  Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$<br><br>"))
            display(self.df.style.hide(axis="index"))
            display(Markdown(f"""<br> The necessary summary Statistis for these data:<br>
                             $$M_A = {{{self.means[0]}}}, M_B = {{{self.means[1]}}}$$
                             $$SS_A = {{{self.ss[0]}}}, SS_B = {{{self.ss[1]}}}$$
                             $$n_A = {{{len(self.df['A'])}}}, n_B = {{{len(self.df['B'])}}}$$<br>
                             """))
        elif self.test == "dependent-samples t-test":
            display(Markdown(f"Given the following within-subjects data, is $M_D$ {text} ${{{self.null}}}$?  Use a ${{{self.tails}}}$ tailed-test with $\\alpha = {{{self.alpha}}}$<br><br>"))
            display(self.df.style.hide(axis="index"))
            display(Markdown(f"""<br>Summary statistics for these data:<br>
                             $$M_A = {{{self.means[0]}}}, M_B = {{{self.means[1]}}}$$
                             $$n = {{{len(self.df['A'])}}}$$ <br>
                             """))
        elif self.test == "one-way ANOVA":
            display(Markdown(f"Given the following between-subjects data, use a one-way ANOVA with $\\alpha = {{{self.alpha}}}$<br><br>"))
            display(self.df.style.hide(axis="index"))
            display(Markdown(f"""<br>Summary statistics for these data:<br>
                             $G = {{{self.g}}} \\quad \\Sigma{{X^2}} = {{{self.sum_squared_scores}}} \\quad k = {{{self.groups}}} \\quad N = {{{self.groups * self.n}}}$ <br>
                             """))
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for group in range(self.groups): # output into a table because the loop seems to onlky work if there is text to start?
                display(Markdown(f"$T_{{{letters[group]}}} = {{{self.sums[group]}}} \\quad SS_{{{letters[group]}}} = {{{self.ss[group]}}}$ <br>"))   
        elif self.test == "repeated-measures ANOVA":
            display(Markdown(f"Given the following within-subjects data, use a repeated-measures ANOVA with $\\alpha = {{{self.alpha}}}$"))
            display(self.df.style.hide(axis="index"))
            display(Markdown(f"""<br>Summary statistics for these data:<br>
                             $G = {{{self.g}}} \\quad \\Sigma X^2 = {{{self.sum_squared_scores}}} \\quad k = {{{self.groups}}} \\quad N = {{{self.groups * self.n}}}$ <br>
                             """))
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for group in range(self.groups):
                display(Markdown(f"$T_{{{letters[group]}}} = {{{self.sums[group]}}} \\quad SS_{{{letters[group]}}} = {{{self.ss[group]}}}$ <br>"))  
        elif self.test == "factorial_ANOVA":
            display(Markdown(f"Given the following data, use a use a 2-Factor ANOVA with $\\alpha = {{{self.alpha}}}$ to test to analyze the following data"))
            #TODO create the summary data table for the factorial ANOVA
            #TODO manage and display the necessay summary statistics in addtion to those presented in the table.
        else:
            return ValueError("test-type specification error in question geneneration")
       

    def set_obt_value(self, value):
        self.obt = value


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
            display(Markdown("The results: <br><br>")) 
            # determine significance
            if self.significance:
                display(Markdown(f"reject the null hypothesis, results are significant, <br> *t*({self.crit_values['degf']}) = {self.obt}, *p* < {self.alpha}, *d* = {self.effect_size}<br><br>"))
            elif not self.significance:
                display(Markdown(f"fail to reject the null hypothesis, results not significant, <br> *t*({self.crit_values['degf']}) = {self.obt}, *p* > {self.alpha}, *d* = {self.effect_size}<br><br>"))
            else:
                return ValueError("significance boolean error in writing results")
        elif self.test == "z":
            # determine significance
            display(Markdown("The results: <br><br>"))
            if self.significance:
                display(Markdown(f"reject the null hypothesis, results are significant, <br> *z* = {self.obt}, *p* < {self.alpha}, *d* = {self.effect_size}<br><br>"))
            elif not self.significance:
                display(Markdown(f"fail to reject the null hypothesis, results not significant, <br> *z* = {self.obt}, *p* > {self.alpha}, *d* = {self.effect_size}<br><br>"))
            else:
                return ValueError("significance boolean error in writing results")
        elif self.test in ["one-way ANOVA", "repeated-measures ANOVA", "factorial_ANOVA"]:
            display(Markdown("The results: <br><br>"))
            if self.significance:
                display(Markdown(f"reject the null hypothesis, results are significant, <br><br> $F({{{self.crit_values['degf_n']}}}, {{{self.crit_values['degf_d']}}}) = {{{self.obt}}}, p < {{{self.alpha}}}, \\eta^2 = {{{self.effect_size}}}$"))
            elif not self.significance:
                display(Markdown(f"fail to reject the null hypothesis, results not significant, <br><br> $F({{{self.crit_values['degf_n']}}}, {{{self.crit_values['degf_d']}}}) = {{{self.obt}}}, p > {{{self.alpha}}}, \\eta^2 = {{{self.effect_size}}}$"))
        else:
            return ValueError("test specificaion error when writing results")


    def write_decision_criteria(self):
        # TODO add more elaborate functionality for the results
        if self.test in ["independent-samples t-test", "one-sample t-test", "dependent-samples t-test"]:
            display(Markdown("<br>The decision criteria: <br><br>"))
            # print the critical value for the test
            if self.tails == 2:
                display(Markdown(f"$t_{{crit}} = \\pm{{{self.crit_values['positive']}}}, \\alpha_{{two-tailed}} = {{{self.alpha}}}, df = {{{self.crit_values['degf']}}}$<br><br>"))
            elif self.tails == 1 and self.crit_values["direction"] == "increase":
                display(Markdown(f"$t_{{crit}} = +{{{self.crit_values['positive']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}, df = {{{self.crit_values['degf']}}}$<br><br>"))
            elif self.tails == 1 and self.crit_values["direction"] == "decrease":
                display(Markdown(f"$t_{{crit}} = {{{self.crit_values['negative']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}, df = {{{self.crit_values['degf']}}}$<br><br>"))
            else:
                return ValueError("tails error in writing results")
        elif self.test == "z":
            # print the critical value of t
            display(Markdown("<br>The decision criteria: <br><br>"))
            if self.tails == 2:
                display(Markdown(f"$z_{{crit}} = \\pm{{{self.crit_values['positive']}}}, \\alpha_{{two-tailed}} = {{{self.alpha}}}$<br><br>"))
            elif self.tails == 1 and self.crit_values["direction"] == "increase":
                display(Markdown(f"$z_{{crit}} = +{{{self.crit_values['positive']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}$<br><br>"))
            elif self.tails == 1 and self.crit_values["direction"] == "decrease":
                display(Markdown(f"$z_{{crit}} = {{{self.crit_values['negative']}}}, \\alpha_{{one-tailed}} = {{{self.alpha}}}$<br><br>"))
            else:
                return ValueError("tails error in writing results")
        elif self.test in ["one-way ANOVA", "repeated-measures ANOVA", "factorial_ANOVA"]:
            display(Markdown("<br> The decision criteria: <br><br>"))
            display(Markdown(f"$F_{{crit}} = {{{self.crit_values['positive']}}}, \\alpha = {{{self.alpha}}}$ <br><br>"))
        else:
            return ValueError("test specificaion error when writing results")

        
    def set_null_hypothesis(self):
        if self.test in ["z", "one-sample t-test"]:
            self.null = self.pop_mean
        else:
            self.null = 0
        return self.null


    def write_hypotheses(self):
        # set the symbols for the hypotheses based on one/two tailed and direction of effect
        if self.tails == 2:
            alt_operation = "\\ne"
            null_operation = "="
        elif self.tails == 1:
            if self.crit_values["direction"] == "increase" :
                alt_operation = "\\gt"
                null_operation = "\\leq"
            elif self.crit_values["direction"] == "decrease":
                alt_operation = "\\lt"
                null_operation = "\\geq"
            else:
                return ValueError("direction error for generating question")
        else:
            return ValueError("tails error when writing hypotheses")

        # use display and markdown to write the null and alternative hypotheses    
        if self.test in ["one-sample t-test", "z"]:
            display(Markdown(f"""State the Hypotheses <br>
                            $$H_0: \\mu {null_operation} {self.null}$$
                            $$H_1: \\mu {alt_operation} {self.null}$$ <br>
                            """))
        elif self.test == "independent-samples t-test":  
            display(Markdown(f"""State the Hypotheses <br>
                            $$H_0: \\mu_A - \\mu_B {null_operation} 0$$
                            $$H_1: \\mu_A - \\mu_B {alt_operation} 0$$ <br>
                            """))
        elif self.test == "dependent-samples t-test":
            display(Markdown(f"""State the Hypotheses <br>
                            $$H_0: \\mu_D {null_operation} {self.null}$$
                            $$H_1: \\mu_D {alt_operation} {self.null}$$ <br>
                            """))
        elif self.test in ["one-way ANOVA", "repeated-measures ANOVA"]:
            pass
        else:
            return ValueError("test specification error in method to write the hypotheses")


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
            self.write_hypotheses()
            self.write_decision_criteria()

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
            self.write_hypotheses()
            self.write_decision_criteria()

            # calculate the standard error
            sem = round(math.sqrt(round((self.var[0]/self.n),2)),2)
            self.obt = round((self.means[0] - self.null) / sem, 2)
            self.effect_size = round((self.means[0] - self.null) / self.std[0], 2)

            # print the caluclations for the standard error
            # TODO add a way to determine environment so output can display in terminal or notebook
            display(Markdown(f"""Calculate the standard error <br>
                            $$s_M = \\sqrt{{\\frac{{s^2}}{{n}}}}$$ <br>
                            $$s_M = \\sqrt{{\\frac{{{self.var[0]}}}{{{self.n}}}}}$$ <br>
                            $$s_M = \\sqrt{{{round((self.var[0]/self.n),2)}}}$$ <br>
                            $$s_M = {{{sem}}}$$ <br><br>
                            calculate $t_{{obt}}$ <br>
                            $$t_{{obt}} = {{\\frac{{M - \\mu}}{{s_M}}}}$$ <br>
                            $$t_{{obt}} = \\frac{{{self.means[0]} - {self.null}}}{{{sem}}}$$ <br>
                            $$t_{{obt}} = \\frac{{{round(self.means[0] - self.null, 2)}}}{{{sem}}}$$ <br>
                            $$t_{{obt}} = {{{self.obt}}}$$ <br><br>
                            calculate Cohen's d <br>
                            $$d = \\frac{{M - \\mu}}{{s}}$$ <br>
                            $$d = \\frac{{{self.means[0]} - {self.null}}}{{{self.std[0]}}}$$ <br>
                            $$d = \\frac{{{round(self.means[0] - self.null, 2)}}}{{{self.std[0]}}}$$ <br>
                            $$d = {{{self.effect_size}}}$$ <br><br>
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
            self.write_hypotheses()
            self.write_decision_criteria()
            
            # primary calculations
            pooled_var = round(((self.ss[0] + self.ss[1]) / ((self.n - 1) + (self.n - 1))), 2)
            sem = round(math.sqrt((round((pooled_var/self.n),2))+(round((pooled_var/self.n),2))),2)
            self.obt = round(((self.means[0] - self.means[1]) - self.null) / sem, 2)
            self.effect_size = round(((self.means[0] - self.means[1])) / round(math.sqrt(pooled_var),2), 2)

            # TODO adapt to display in the terminal or a notebook
            # display the caluclations for the pooled variance
            display(Markdown(f"""Calculate the pooled variance: <br>
                            $$s_p^2 = {{\\frac{{SS_A + SS_B}}{{df_A + df_B}}}}$$ <br>
                            $$s_p^2 = {{\\frac{{{self.ss[0]} + {self.ss[1]}}}{{{self.n - 1} + {self.n - 1}}}}}$$ <br>
                            $$s_p^2 = \\frac{{{round(self.ss[0] + self.ss[1],2)}}}{{{(self.n - 1) + (self.n - 1)}}}$$ <br>
                            $$s_p^2 = {{{pooled_var}}}$$ <br><br>
                            """))
            # display the calculations for the estimated standard error
            display(Markdown(f"""Calculate the estimated error of the difference between means: $s_{{(M_A - M_B)}}$ <br>
                            $$s_{{(M_A - M_B)}} = \\sqrt{{\\frac{{s_p^2}}{{n_1}} + \\frac{{s_p^2}}{{n_1}}}}$$ <br>
                            $$s_{{(M_A - M_B)}} = \\sqrt{{\\frac{{{pooled_var}}}{{{self.n}}} + \\frac{{{pooled_var}}}{{{self.n}}}}}$$ <br>
                            $$s_{{(M_A - M_B)}} = \\sqrt{{{round(pooled_var/self.n, 2)} + {round(pooled_var/self.n, 2)}}}$$ <br>
                            $$s_{{(M_A - M_B)}} = \\sqrt{{{round(pooled_var/self.n, 2) + round(pooled_var/self.n, 2)}}}$$ <br>
                            $$s_{{(M_A - M_B)}} = {{{sem}}}$$ <br><br>
                            """))
            # display the caluclations for t_obt
            display(Markdown(f"""Calculate $t_{{obt}}$ <br>
                            $$t_{{obt}} = {{\\frac{{(M_A - M_B) - (\\mu_A - \\mu_B)}}{{s_{{(M_A - M_B)}}}}}}$$ <br>
                            $$t_{{obt}} = \\frac{{({self.means[0]} - {self.means[1]}) - {{{self.null}}}}}{{{sem}}}$$ <br>
                            $$t_{{obt}} = \\frac{{{round(self.means[0] - self.means[1] - self.null, 2)}}}{{{sem}}}$$ <br>
                            $$t_{{obt}} = {{{self.obt}}}$$ <br><br>
                            """))
            # print calculations for cohen's d
            display(Markdown(f"""Calculate Cohen's *d*  <br>
                            $$d = \\frac{{M_A - M_B}}{{\\sqrt{{s_p^2}}}}$$ <br>
                            $$d = \\frac{{({self.means[0]} - {self.means[1]})}}{{{{{{\\sqrt{{{pooled_var}}}}}}}}}$$ <br>
                            $$d = \\frac{{({round(self.means[0] - self.means[1], 2)})}}{{{round(math.sqrt(pooled_var),2)}}}$$ <br>
                            $$d = {{{self.effect_size}}}$$ <br><br>
                            """))

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
            self.write_hypotheses()
            self.write_decision_criteria()

            # primary calculations
            # need to gather the difference scores
            self.df['D'] = self.df['B'] - self.df['A']
            
            # print the dataframe with the difference scores
            display(Markdown("Calculating the difference scores $D = X_B - X_A$ <br>"))
            display(self.df.style.hide(axis="index"))

            # Calculate the Mean of the Difference Scores
            sum_d = self.df['D'].sum()
            n = len(self.df['D'])
            mean_d = round(sum_d/n, 2)
            display(Markdown(f"""<br>Calculate the Mean of the Difference Scores
                            $$M_D = \\frac{{\\Sigma D}}{{n}}$$ <br>
                            $$M_D = \\frac{{{sum_d}}}{{{n}}}$$ <br>
                            $$M_D = {{{mean_d}}}$$ <br><br>
                            """))
  
            # calculate the SS for the difference scores
            self.df['D^2'] = self.df['D'].apply(lambda x: x ** 2)
            sum_sqared_scores = self.df['D^2'].sum()
            ss = round(sum_sqared_scores - round((sum_d ** 2)/n, 2), 2)
            
            # print the dataframe with the squared difference scores
            display(Markdown("Create a column for the squared difference scores $D^2$"))
            display(self.df.style.hide(axis="index"))

            # summary stats for the difference scores
            display(Markdown(f"""<br>New Summary statistics for the difference scores:<br>
                             $$M_D = {{{mean_d}}}, \\quad \\Sigma{{D}} = {{{sum_d}}}, \\quad \\Sigma{{D^2}} = {{{sum_sqared_scores}}}$$ <br>
                             """))

            display(Markdown(f"""<br>Calculate *SS* of the difference scores <br>
                             $$SS_D = \\Sigma D^2 - \\frac{{(\\Sigma D)^2}}{{n}}$$ <br>
                             $$SS_D = {{{sum_sqared_scores}}} - \\frac{{{sum_d ** 2}}}{{{n}}}$$ <br>
                             $$SS_D = {{{sum_sqared_scores}}} - {{{round((sum_d ** 2)/n, 2)}}}$$ <br>
                             $$SS_D = {{{ss}}}$$ <br><br>
                             """))
            
            # calculate the variance    
            variance = round(ss / (n - 1), 2)
            display(Markdown(f"""Calculate the variance of the difference scores <br>
                             $$s_D^2 = \\frac{{SS_D}}{{df}}$$ <br>
                             $$s_D^2 = \\frac{{{ss}}}{{{n - 1}}}$$ <br>
                             $$s_D^2 = {{{variance}}}$$ <br><br>
                            """))

            # Calculate the estimated standard error
            sem = round(math.sqrt(variance/n), 2)
            display(Markdown(f"""Calculate the estimated standard error of the difference scores <br>
                             $$s_{{M_D}} = \\sqrt{{\\frac{{s^2}}{{n}}}}$$ <br>
                             $$s_{{M_D}} = \\sqrt{{\\frac{{{variance}}}{{{n}}}}}$$ <br>
                             $$s_{{M_D}} = \\sqrt{{{round(variance/n, 2)}}}$$ <br>
                             $$s_{{M_D}} = {{{sem}}}$$ <br><br>
                            """))

            # caclulate the t-statistic
            self.obt = round((mean_d - self.null) / sem, 2)
            display(Markdown(f"""Calculate $t_{{obt}}$ <br>
                             $$t_{{obt}} = {{\\frac{{M_D - \\mu_D}}{{s_{{M_D}}}}}}$$ <br>
                             $$t_{{obt}} = \\frac{{{mean_d} - {self.null}}}{{{sem}}}$$ <br>
                             $$t_{{obt}} = \\frac{{{round(mean_d - self.null, 2)}}}{{{sem}}}$$ <br>
                             $$t_{{obt}} = {{{self.obt}}}$$ <br><br>
                            """))

            # print calculations for cohen's d
            self.effect_size = round(mean_d / round(math.sqrt(variance),2), 2)
            display(Markdown(f"""Calculating Cohen's *d* <br>
                             $$d = \\frac{{M_D}}{{\\sqrt{{s^2}}}}$$ <br>
                             $$d = \\frac{{{mean_d}}}{{{{{{\\sqrt{{{variance}}}}}}}}}$$ <br>
                             $$d = \\frac{{{mean_d}}}{{{round(math.sqrt(variance),2)}}}$$ <br>
                             $$d = {{{self.effect_size}}}$$ <br><br>
                            """))

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
            self.write_hypotheses()
            self.write_decision_criteria()

            # Primary Calculations
            big_n = self.groups * self.n
            
            # degrees of freedom
            df_total = big_n - 1
            df_between = self.groups - 1
            df_within = big_n - self.groups

            print() # blank space
            if test == "repeated-measures":
                display(Markdown("""Stage 1 Calculations: <br><br>"""))
            
            display(Markdown(f"""Calculate the Degrees of Freedom <br>
                    $$df_{{total}} = N - 1$$ <br>
                    $$df_{{total}} = {{{big_n}}} - 1$$ <br>
                    $$df_{{total}} = {{{df_total}}}$$ <br><br>
                    $$df_{{between}} = k - 1$$ <br>
                    $$df_{{between}} = {{{self.groups}}} - 1$$ <br>
                    $$df_{{between}} = {{{df_between}}}$$ <br><br>
                    $$df_{{within}} = N - K$$ <br>
                    $$df_{{within}} = {{{big_n}}} - {{{self.groups}}}$$ <br>
                    $$df_{{within}} = {{{df_within}}}$$ <br><br>
                """))

            # sum of squares
            ss_total = self.sum_squared_scores - round(((self.g**2)/big_n), 2)
            ss_within = 0
            for group in range(self.groups):
                ss_within += self.ss[group]
            ss_between = ss_total - ss_within

            values = ""
            for group in range(self.groups):
                if group == 0:
                    values += f"{self.ss[group]}"
                else:
                    values += f" + {self.ss[group]}"
            
            #TODO consider writing methods to take the values below to display each piece seperately.
            display(Markdown(f"""Calculate the Sum of Squares <br>
                    $$SS_{{total}} = \\Sigma X^2 - \\frac{{G^2}}{{N}}$$ <br>
                    $$SS_{{total}} = {{{self.sum_squared_scores}}} - \\frac{{{self.g}^2}}{{{big_n}}}$$ <br>
                    $$SS_{{total}} = {{{self.sum_squared_scores}}} - \\frac{{{self.g**2}}}{{{big_n}}}$$ <br>
                    $$SS_{{total}} = {{{self.sum_squared_scores}}} - {{{round(((self.g**2)/big_n), 2)}}}$$ <br>
                    $$SS_{{total}} = {{{round(ss_total, 2)}}}$$ <br><br>
                    $$SS_{{within}} = \\Sigma SS_{{inside\\_each\\_condition}}$$ <br>
                    $$SS_{{within}} = {{{values}}}$$ <br>
                    $$SS_{{within}} = {{{round(ss_within, 2)}}}$$ <br><br>
                    $$SS_{{between}} = SS_{{total}} - SS_{{within}}$$ <br>
                    $$SS_{{between}} = {{{round(ss_total, 2)}}} - {{{round(ss_within, 2)}}}$$ <br>
                    $$SS_{{between}} = {{{round(ss_between, 2)}}}$$ <br><br>
                    note: the other way to calculate $SS_{{betwen}}$ is: <br>
                    $$SS_{{between}} = \\Sigma{{\\frac{{T^2}}{{n}}}} - \\frac{{G^2}}{{N}}$$ <br><br>
                """))

            if self.test == "repeated-measures ANOVA":
                # degrees of freedom
                df_subjects = self.n - 1
                df_error = df_within - df_subjects
                

                display(Markdown(f"""Stage 2 Calculations: <br><br>
                    Partition the Degrees of Freedom <br>
                    $$df_{{subjects}} = n - 1$$ <br>
                    $$df_{{subjects}} = {{{self.n}}} - 1$$ <br>
                    $$df_{{subjects}} = {{{df_subjects}}}$$ <br><br>
                    $$df_{{error}} = df_{{within}} - df_{{subjects}}$$ <br>
                    $$df_{{error}} = {{{df_within}}} - {{{df_subjects}}}$$ <br>
                    $$df_{{error}} = {{{df_error}}}$$ <br><br>
                """))
                
                display(Markdown("Calculate person sums and add a new column (*P*)"))
                self.df["P"] = self.df.drop(columns=['ID']).sum(axis = 1)
                display(self.df.style.hide(axis="index"))
                print()

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

                display(Markdown(f"$$SS_{{subjects}} = \\Sigma{{\\frac{{P^2}}{{k}}}} - \\frac{{G^2}}{{N}}$$ <br>"))
                temp_text = ""
                for p in p_sums:
                    temp_text += f" + \\frac{{{p}^2}}{{{self.groups}}}"
                display(Markdown(f"$$SS_{{subjects}} = {{{temp_text[3:]}}} - \\frac{{{self.g}^2}}{{{big_n}}}$$ <br>"))
                temp_text = ""
                for p in p_squared:
                    temp_text += f" + \\frac{{{p}}}{{{self.groups}}}"
                display(Markdown(f"$$SS_{{subjects}} = {{{temp_text[3:]}}} - \\frac{{{self.g**2}}}{{{big_n}}}$$ <br>"))
                temp_text = ""
                for p in quotients:
                    temp_text += f" + {{{p}}}"

                display(Markdown(f"""$$SS_{{subjects}} = {{{temp_text[3:]}}} - {{{round(((self.g**2)/big_n), 2)}}}$$ <br>
                    $$SS_{{subjects}} = {{{round(sum_quotients,2)}}} - {{{round(((self.g**2)/big_n), 2)}}}$$ <br>
                    $$SS_{{subjects}} = {{{round(ss_subjects,2)}}}$$ <br><br>
                    $$SS_{{error}} = SS_{{within}} - SS_{{subjects}}$$ <br>
                    $$SS_{{error}} = {{{round(ss_within,2)}}} - {{{round(ss_subjects,2)}}}$$ <br>
                    $$SS_{{error}} = {{{round(ss_error,2)}}}$$ <br><br>
                """))

            # mean squares
            ms_between = round(round(ss_between, 2)/df_between, 2)
            ms_within = round(round(ss_within, 2)/df_within, 2)

            display(Markdown(f"""Calculate the Mean Squares <br>
                $$MS_{{between}} = \\frac{{SS_{{between}}}}{{df_{{between}}}}$$ <br>
                $$MS_{{between}} = \\frac{{{round(ss_between, 2)}}}{{{df_between}}}$$ <br>
                $$MS_{{between}} = {{{round(ms_between, 2)}}}$$ <br><br>
            """))

            if self.test == "one-way ANOVA":
                display(Markdown(f"""$$MS_{{within}} = \\frac{{SS_{{within}}}}{{df_{{within}}}}$$ <br>
                    $$MS_{{within}} = \\frac{{{round(ss_within, 2)}}}{{{df_within}}}$$ <br>
                    $$MS_{{within}} = {{{round(ms_within, 2)}}}$$ <br><br>
                """))
                
                # F value
                self.obt = round(ms_between/ms_within, 2)

                display(Markdown(f"""Calculate the *F*-Ratio <br>
                    $$F_{{obt}} = \\frac{{MS_{{between}}}}{{MS_{{within}}}}$$ <br>
                    $$F_{{obt}} = \\frac{{{round(ms_between,2)}}}{{{round(ms_within,2)}}}$$ <br>
                    $$F_{{obt}} = {{{round(self.obt, 2)}}}$$ <br><br>
                """))

                # effect size
                self.effect_size = round(round(ss_between, 2)/round(ss_total, 2), 2)
                
                display(Markdown(f"""Calculate $\\eta^2$ <br>
                    $$\\eta^2 = \\frac{{SS{{between}}}}{{SS_{{total}}}}$$ <br>
                    $$\\eta^2 = \\frac{{{round(ss_between, 2)}}}{{{round(ss_total, 2)}}}$$ <br>
                    $$\\eta^2 = {{{round(self.effect_size, 2)}}}$$ <br><br>
                """))

            elif test == "repeated-measures":
                ms_error = round(round(ss_error,2)/df_error, 2) # type: ignore

                display(Markdown(f"""$$MS_{{error}} = \\frac{{SS_{{error}}}}{{df_{{error}}}}$$ <br>
                    $$MS_{{error}} = \\frac{{{round(ss_error, 2)}}}{{{df_error}}}$$ <br> 
                    $$MS_{{error}} = {{{round(ms_error, 2)}}}$$ <br><br>
                """)) # type: ignore

                # F value
                self.obt = round(ms_between/ms_error, 2)

                display(Markdown(f"""Calculate the *F*-Ratio <br>
                    $$F_{{obt}} = \\frac{{MS_{{between}}}}{{MS_{{error}}}}$$ <br>
                    $$F_{{obt}} = \\frac{{{round(ms_between,2)}}}{{{round(ms_error,2)}}}$$ <br>
                    $$F_{{obt}} = {{{round(self.obt, 2)}}}$$ <br><br>
                """))

                # effect size
                self.effect_size = round(round(ss_between, 2)/(round(ss_total, 2) - round(ss_subjects, 2)), 2) # type: ignore
                
                display(Markdown(f"""Calculate $\\eta^2$ <br>
                    $$\\eta_p^2 = \\frac{{SS{{between}}}}{{SS_{{total}} - SS_{{subjects}}}}$$ <br>
                    $$\\eta_p^2 = \\frac{{{round(ss_between, 2)}}}{{{{{round(ss_total, 2)}}} - {{{round(ss_subjects, 2)}}}}}$$ <br>
                    $$\\eta_p^2 = \\frac{{{round(ss_between, 2)}}}{{{round(ss_total, 2) - round(ss_subjects, 2)}}}$$ <br> 
                    $$\\eta_p^2 = {{{round(self.effect_size, 2)}}}$$ <br><br>
                """)) # type: ignore               

            else:
                raise ValueError("test accepts: 'one-way', 'repeated-measures'")

            self.significance = self.final_decision()
            self.write_result()
            print() # blank space

"""
    def generate_factorial_data(self, design: tuple):
        # 1. Define Factors and Levels
        factor_a = [f'A{level+1}' for level in range(design[0])]
        factor_b = [f'B{level+1}' for level in range(design[1])]
        n = self.n # Number of samples per combination

        # 2. Generate all combinations (e.g., 2x3 = 6 combinations)
        combinations = list(itertools.product(factor_a, factor_b))

        # 3. Define distribution parameters for each combination
        # Mean, StdDev for each (A1B1, A1B2, A1B3, A2B1, A2B2, A2B3)
        parameters = []
        for condition in combinations:
            mean = self.pop_mean
            sd = self.pop_sd
            same_diff = random.randint(0,3)
            if same_diff >= 1:
                effect =  round(mean * random.uniform(0.10, 0.50))
                mean +=  effect
            parameters.append((mean, sd))

        # 4. Generate data
        data = []
        for i, (a, b) in enumerate(combinations):
            # Draw random samples from normal distribution using params[i]
            values = np.random.normal(loc=parameters[i][0], scale=parameters[i][1], size=n).astype(int)
            for val in values:
                data.append({'Factor_A': a, 'Factor_B': b, 'X': val})

        # 5. Create DataFrame
        df = pd.DataFrame(data)

        return df


    def factorial_ANOVA(self, design: tuple):
        self.factorial_data = {}
        self.test = "factorial_ANOVA"
        self.df = self.generate_factorial_data(design)

        
        # Overall Values
        grand_sum = self.df['X'].sum()
        grand_n = self.df['X'].count()
        self.df['sum_squared'] = self.df['X'].apply(lambda x: x ** 2)
        grand_sum_squared = self.df['sum_squared'].sum()
        grand_ss = round(grand_sum_squared - round((grand_sum ** 2)/grand_n, 2),2)
        

        # Factor A
        sums_factor_a = self.df.groupby('Factor_A')['X'].sum()
        sum_sq_fa = self.df.groupby('Factor_A')['sum_squared'].sum()
        n_factor_a = self.df.groupby('Factor_A')['X'].count()
        self.factorial_data["factor_A"] = [(x,y,z) for (x,y,z) in zip()]

        ss_fa = []
        for sum, ss, n in zip(sums_factor_a, sum_sq_fa, n_factor_a):
            ss_vals = ss - round((sum ** 2)/n, 2)
            ss_fa.append(round(ss_vals, 2))

        # Factor B
        sums_factor_b = self.df.groupby('Factor_B')['X'].sum()
        sum_sq_fb = self.df.groupby('Factor_B')['sum_squared'].sum()
        n_factor_b = self.df.groupby('Factor_B')['X'].count()
        
        ss_fb = []
        for sum, ss, n in zip(sums_factor_b, sum_sq_fb, n_factor_b):
            ss_vals = ss - round((sum ** 2)/n, 2)
            ss_fb.append(round(ss_vals, 2))

        print(grand_ss, ss_fa, ss_fb)
"""

class FactorialData:
    def __init__(self, design: tuple, group_n: int):
        self.n = group_n
        self.base = RandomData(n = group_n)
        self.design = design
        self.df = self.generate_factorial_data()
        self.summary = {}
        self.final_calculations = {}


    def generate_factorial_data(self):
        # 1. Define Factors and Levels
        factor_a = [f'A{level+1}' for level in range(self.design[0])]
        factor_b = [f'B{level+1}' for level in range(self.design[1])]
        n = self.base.n # Number of samples per combination

        # 2. Generate all combinations (e.g., 2x3 = 6 combinations)
        conditions = list(itertools.product(factor_a, factor_b))
        self.condition_list = []
        for a in factor_a:
            for b in factor_b:
                self.condition_list.append(f"{a}{b}")
        
        # 3. Define distribution parameters for each combination
        # Mean, StdDev for each (A1B1, A1B2, A1B3, A2B1, A2B2, A2B3)
        parameters = []
        for condition in conditions:
            mean = self.base.pop_mean
            sd = self.base.pop_sd
            same_diff = random.randint(0,3)
            if same_diff >= 1:
                effect =  round(mean * random.uniform(0.10, 0.50))
                mean +=  effect
            parameters.append((mean, sd))

        # 4. Generate data
        data = []
        for i, (a, b) in enumerate(conditions):
            # Draw random samples from normal distribution using params[i]
            values = np.random.normal(loc=parameters[i][0], scale=parameters[i][1], size=n).astype(int)
            for val in values:
                data.append({'Factor_A': a, 'Factor_B': b, 'X': val})

        # 5. Create DataFrame
        df = pd.DataFrame(data)

        return df


    def combined_values(self):
        grand_sum = self.df['X'].sum()
        grand_n = self.df['X'].count()
        self.df['sum_squared'] = self.df['X'].apply(lambda x: x ** 2)
        grand_sum_squared = self.df['sum_squared'].sum()
        grand_ss = round(grand_sum_squared - round((grand_sum ** 2)/grand_n, 2),2)
        self.summary.update({"grand_ss": grand_ss, 
                             "total_n": grand_n, 
                             "grand_sum_scores": grand_sum,
                             "grand_sum_squared_scores": grand_sum_squared})


    def factor_values(self):
        for factor in ["Factor_A", "Factor_B"]:
            self.summary.update({
                f"levels_{factor}": len(self.df.groupby(factor)['X'].count()),
                f"x_sums_{factor}": [i for i in self.df.groupby(factor)['X'].sum()],
                f"x_sq_sum_{factor}": [i for i in self.df.groupby(factor)['sum_squared'].sum()],
                f"n_{factor}": [i for i in self.df.groupby(factor)['X'].count()]
            })
            means = []
            ss_values = []
            for sum, ss, n in zip(self.summary[f"x_sums_{factor}"], self.summary[f"x_sq_sum_{factor}"], self.summary[f"n_{factor}"]):
                ss_vals = ss - round((sum ** 2)/n, 2)
                ss_values.append(round(ss_vals, 2))
                means.append(round(sum / n, 2))
            self.summary[f"ss_values_{factor}"] = ss_values
            self.summary[f"means_{factor}"] = means

    
    def summary_by_group(self):
        t = [i for i in self.df.groupby(['Factor_A', 'Factor_B'])['X'].sum()]
        sum_sqs = [i for i in self.df.groupby(['Factor_A', 'Factor_B'])['sum_squared'].sum()]
        ns = [i for i in self.df.groupby(['Factor_A', 'Factor_B'])['X'].count()]
        ss_values = []
        means = []
        for sum, ss, n in zip(t, sum_sqs, ns):
            ss_vals = ss - round((sum ** 2)/n, 2)
            ss_values.append(round(ss_vals, 2))
            means.append(round(sum / n, 2))

        group_totals = {f'T_{level}': t for level, t in zip(self.condition_list, t)}
        #group_sums = {f'sum_sq_{level}': ss for level, ss in zip(self.condition_list, sum_sqs)}
        group_ns = {f'n_{level}': n for level, n in zip(self.condition_list, ns)}
        self.group_ss = {f'SS_{level}': ss for level, ss in zip(self.condition_list, ss_values)}
        group_means = {f'M_{level}': m for level, m in zip(self.condition_list, means)}

        #TODO format into a table for problem display
        for condition in self.condition_list:
            display(Markdown(f"$T_{{{condition}}} = {{{group_totals[f"T_{condition}"]}}} \\quad M_{{{condition}}} = {{{group_means[f"M_{condition}"]}}}\\quad SS_{{{condition}}} = {{{self.group_ss[f"SS_{condition}"]}}} \\quad n_{{{condition}}} = {{{group_ns[f"n_{condition}"]}}}$ <br>"))    
        
    
    def stage_1_ss(self):
        display(Markdown("Stage 1 SS calculations"))
        # sum of squares
        ss_total = self.summary['grand_sum_squared_scores'] - round((self.summary["grand_sum_scores"] ** 2) / self.summary["total_n"], 2)
        ss_within = 0
        display_values = ""
        for i, value in enumerate(self.group_ss.values()):
            ss_within += value
            if i == 0:
                display_values += f"{value}"
            else:
                display_values += f" + {value}"
        ss_between = ss_total - ss_within
        self.final_calculations.update({
            "SS_Total": round(ss_total, 2),
            "SS_Between": round(ss_between, 2),
            "SS_Within": round(ss_within, 2)
        })
        
        #TODO consider writing methods to take the values below to display each piece seperately.
        display(Markdown(f"""Calculate the Sum of Squares <br><br>
                $SS_{{total}} = \\Sigma X^2 - \\frac{{G^2}}{{N}}$ <br><br>
                $SS_{{total}} = {{{self.summary['grand_sum_squared_scores']}}} - \\frac{{{self.summary["grand_sum_scores"]}^2}}{{{self.summary["total_n"]}}}$ <br><br>
                $SS_{{total}} = {{{self.summary['grand_sum_squared_scores']}}} - \\frac{{{self.summary["grand_sum_scores"] ** 2}}}{{{self.summary["total_n"]}}}$ <br><br>
                $SS_{{total}} = {{{self.summary['grand_sum_squared_scores']}}} - {{{round((self.summary["grand_sum_scores"] ** 2) / self.summary["total_n"], 2)}}}$ <br><br>
                $SS_{{total}} = {{{round(ss_total, 2)}}}$ <br><br><br>
                $SS_{{within}} = \\Sigma SS_{{inside\\_each\\_condition}}$ <br><br>
                $SS_{{within}} = {{{display_values}}}$ <br><br>
                $SS_{{within}} = {{{round(ss_within, 2)}}}$ <br><br><br>
                $SS_{{between}} = SS_{{total}} - SS_{{within}}$ <br><br>
                $SS_{{between}} = {{{round(ss_total, 2)}}} - {{{round(ss_within, 2)}}}$ <br><br>
                $SS_{{between}} = {{{round(ss_between, 2)}}}$ <br><br><br>
                note: the other way to calculate $SS_{{betwen}}$ is: <br><br>
                $SS_{{between}} = \\Sigma{{\\frac{{T^2}}{{n}}}} - \\frac{{G^2}}{{N}}$ <br><br><br>
            """))
        

    def stage_1_df(self):
        big_n = self.summary["total_n"]
        df_total = big_n - 1
        conditions = len(self.condition_list)
        df_between = conditions - 1
        df_within = big_n - conditions
        display(Markdown(f"""Calculate the Degrees of Freedom <br><br>
                    $df_{{total}} = N - 1$ <br><br>
                    $df_{{total}} = {{{big_n}}} - 1$ <br><br>
                    $df_{{total}} = {{{df_total}}}$ <br><br><br>
                    $df_{{between}} = k - 1$ <br><br>
                    $df_{{between}} = {{{conditions}}} - 1$ <br><br>
                    $df_{{between}} = {{{df_between}}}$ <br><br><br>
                    $df_{{within}} = N - K$ <br><br>
                    $df_{{within}} = {{{big_n}}} - {{{conditions}}}$ <br><br>
                    $df_{{within}} = {{{df_within}}}$ <br><br><br>
                """))
        self.final_calculations.update({
            "df_total": df_total,
            "df_between": df_between,
            "df_within": df_within
        })

        
    def collapse_by_factor(self, factor: str):
        levels = [f"{factor[-1]}_{i + 1}" for i in range(len(self.summary[f'n_{factor}']))]
        display(Markdown(f"Summary Data for {factor} <br>"))
        for i, level in enumerate(levels):
            display(Markdown(f"$T_{{{level}}} = {{{self.summary[f"x_sums_{factor}"][i]}}}  \\quad n_{{{level}}} = {{{self.summary[f"n_{factor}"][i]}}}$ <br>")) 
            # unused display calculations: \\quad M_{{{level}}} = {{{self.summary[f"means_{factor}"][i]}}}\\quad SS_{{{level}}} = {{{self.summary[f"ss_values_{factor}"][i]}}}


    def partition_ss(self):
        #display(Markdown(f"Partition the SS for ${factor}$"))
        for factor in ["Factor_A", "Factor_B"]:
            levels = [f"{factor[-1]}_{i + 1}" for i in range(len(self.summary[f'n_{factor}']))]
            # step 4 sum the factor components
            result = 0
            for i, level in enumerate(levels):
                result += round((self.summary[f"x_sums_{factor}"][i] ** 2) / self.summary[f"n_{factor}"][i], 2)
            # step 5 subtract the component from the full data
            result -= round((self.summary["grand_sum_scores"] ** 2) / self.summary["total_n"], 2)
            self.final_calculations[f"SS_{factor}"] = round(result, 2)   

        # the interaction
        ss_interaction = round(self.final_calculations["SS_Between"] - self.final_calculations["SS_Factor_A"] - self.final_calculations["SS_Factor_B"], 2)
        self.final_calculations["SS_AxB"] = ss_interaction
  

    def display_ss(self, factor: str):
        display(Markdown(f"Partition the SS for ${factor}$"))
        if factor in ["Factor_A", "Factor_B"]:
            levels = [f"{factor[-1]}_{i + 1}" for i in range(len(self.summary[f'n_{factor}']))]
            display(Markdown(f"Calculate $SS_{{{factor}}}$ <br><br>"))
            display(Markdown(f"$\\Sigma{{\\frac{{{{T_{{{factor}}}}}^2}}{{n_{{{factor}}}}}}} - \\frac{{G^2}}{{N}}$ <br><br>"))
            equation_text = ""
            for i, level in enumerate(levels):
                if i == 0:
                    equation_text += f"\\frac{{{{{self.summary[f"x_sums_{factor}"][i]}}}^2}}{{{self.summary[f"n_{factor}"][i]}}}"
                else:
                    equation_text += f" + \\frac{{{{{self.summary[f"x_sums_{factor}"][i]}}}^2}}{{{self.summary[f"n_{factor}"][i]}}}"
            display(Markdown(f"$SS_{{{factor}}} = {{{equation_text}}} - \\frac{{{{{self.summary["grand_sum_scores"]}}}^2}}{{{self.summary["total_n"]}}}$ <br>"))
            
            # step 2 in the calculations display the squared values in the numerator
            equation_text = ""
            for i, level in enumerate(levels):
                if i == 0:
                    equation_text += f"\\frac{{{{{self.summary[f"x_sums_{factor}"][i] ** 2}}}}}{{{self.summary[f"n_{factor}"][i]}}}"
                else:
                    equation_text += f" + \\frac{{{{{self.summary[f"x_sums_{factor}"][i] ** 2}}}}}{{{self.summary[f"n_{factor}"][i]}}}"
            display(Markdown(f"$SS_{{{factor}}} = {{{equation_text}}} - \\frac{{{{{self.summary["grand_sum_scores"] ** 2}}}}}{{{self.summary["total_n"]}}}$"))
            
            # step 3 in the calculations display results of division
            equation_text = ""
            for i, level in enumerate(levels):
                if i == 0:
                    equation_text += f"{{{round((self.summary[f"x_sums_{factor}"][i] ** 2) / self.summary[f"n_{factor}"][i], 2)}}}"
                else:
                    equation_text += f" + {{{round((self.summary[f"x_sums_{factor}"][i] ** 2) / self.summary[f"n_{factor}"][i], 2)}}}"
            display(Markdown(f"$SS_{{{factor}}} = {{{equation_text}}} - {{{round((self.summary["grand_sum_scores"] ** 2) / self.summary["total_n"], 2)}}}$ <br>"))

            # step 4 sum the factor components
            result = 0
            for i, level in enumerate(levels):
                result += round((self.summary[f"x_sums_{factor}"][i] ** 2) / self.summary[f"n_{factor}"][i], 2)
            display(Markdown(f"$SS_{{{factor}}} = {{{round(result, 2)}}} - {{{round((self.summary["grand_sum_scores"] ** 2) / self.summary["total_n"], 2)}}}$ <br>"))

            # step 5 subtract the component from the full data
            display(Markdown(f"$SS_{{{factor}}} = {{{round(self.final_calculations[f"SS_{factor}"], 2)}}}$ <br>"))
        else:
            display(Markdown(f"Calculate $SS_{{AxB}}$ <br>"))
            display(Markdown("$SS_{{AxB}} = SS_{{Between}} - SS_{{Factor_A}} - SS_{{Factor_B}}$<br>"))
            display(Markdown(f"$SS_{{AxB}} = {{{self.final_calculations["SS_Between"]}}} - {{{self.final_calculations["SS_Factor_A"]}}} - {{{self.final_calculations["SS_Factor_B"]}}}$ <br>"))
            display(Markdown(f"$SS_{{AxB}} = {{{self.final_calculations["SS_AxB"]}}}$ <br>"))


    def partition_df(self):
        levels_a = self.summary["levels_Factor_A"]
        levels_b = self.summary["levels_Factor_B"]
        
        df_a = levels_a - 1
        df_b = levels_b - 1
        df_axb = self.final_calculations["df_between"] - df_a - df_b

        self.final_calculations.update({
            "df_Factor_A": df_a,
            "df_Factor_B": df_b,
            "df_AxB": df_axb
        })


    def display_df(self, factor: str):
        if factor in ["Factor_A", "Factor_B"]:
            display(Markdown(f"""partition df for ${factor}$ <br><br>
                             $df_{{{factor}}} = k_{{{factor}}} - 1$ <br><br>
                             $df_{{{factor}}} = {{{self.summary[f"levels_{factor}"]}}} - 1$ <br><br>
                             $df_{{{factor}}} = {{{self.final_calculations[f"df_{factor}"]}}}$ <br><br>"""))

        else:
            display(Markdown(f"""Partition df for the interaction <br><br>
                            $df_{{AxB}} = df_{{Between}} -df_{{Factor_A}} - df_{{Factor_B}}$ <br><br>
                            $df_{{AxB}} = {{{self.final_calculations["df_between"]}}} - {{{self.final_calculations["df_Factor_A"]}}} - {{{self.final_calculations["df_Factor_B"]}}}$ <br><br>
                            $df_{{AxB}} = {{{self.final_calculations["df_AxB"]}}}$ <br><br><br>
                             """))
        
    
    def mean_squares(self):
        ms_within = round(self.final_calculations["SS_Within"] / self.final_calculations["df_within"], 2)
        ms_a = round(self.final_calculations["SS_Factor_A"] / self.final_calculations["df_Factor_A"], 2)
        ms_b = round(self.final_calculations["SS_Factor_B"] / self.final_calculations["df_Factor_B"], 2)
        ms_axb = round(self.final_calculations["SS_AxB"] / self.final_calculations["df_AxB"], 2)

        self.final_calculations.update({
            "ms_within": ms_within,
            "ms_Factor_A": ms_a,
            "ms_Factor_B": ms_b,
            "ms_AxB": ms_axb
        })


    def display_ms(self, factor: str):
        ss = self.final_calculations[f"SS_{factor}"]
        df = self.final_calculations[f"df_{factor}"]
        ms = self.final_calculations[f"ms_{factor}"]
        display(Markdown(f"""Mean Squares for ${{{factor}}}$ <br><br>
                        $MS_{{{factor}}} = \\frac{{SS_{{{factor}}}}}{{df_{{{factor}}}}}$<br> <br>
                        $MS_{{{factor}}} = \\frac{{{ss}}}{{{df}}}$ <br><br>
                        $MS_{{{factor}}} = {{{ms}}}$<br><br>
                         """))


    def f_ratios(self):
        f_a = round(self.final_calculations["ms_Factor_A"] / self.final_calculations["ms_within"], 2)
        f_b = round(self.final_calculations["ms_Factor_B"] / self.final_calculations["ms_within"], 2)
        f_axb = round(self.final_calculations["ms_AxB"] / self.final_calculations["ms_within"], 2)

        self.final_calculations.update({
            "f_Factor_A": f_a,
            "f_Factor_B": f_b,
            "f_AxB": f_axb
        })


    def display_f_ratios(self, factor: str):
        numerator = self.final_calculations[f"ms_{factor}"]
        denominator = self.final_calculations["ms_within"]
        f_ratio = self.final_calculations[f"f_{factor}"]
        self.base.set_obt_value(f_ratio)
        display(Markdown(f"""Calculate the F-Ratio for ${{{factor}}}$ <br><br>
                        $F_{{{factor}}} = \\frac{{MS_{{{factor}}}}}{{MS_{{Within}}}}$ <br><br>
                        $F_{{{factor}}} = \\frac{{{numerator}}}{{{denominator}}}$ <br><br>
                        $F_{{{factor}}} = {{{f_ratio}}}$ <br><br>
                        """))


    def test_crit_values(self, factor: str):
            degf_d = self.final_calculations["df_within"]
            degf_n = self.final_calculations[f"df_{factor}"]
            crit = round(stats.f.ppf(q = (1 - self.base.alpha), dfn = degf_n, dfd = degf_d), 2) # type: ignore
            self.base.set_crit_values(crit, degf_n = degf_n, degf_d = degf_d)


    def factorial_ANOVA(self):
        self.base.set_test("factorial_ANOVA")
        self.factorial_data = {}
        self.combined_values()
        self.factor_values()

        self.base.generate_question()
        self.base.set_null_hypothesis()

        display(Markdown("Full Group Summary Data"))
        self.summary_by_group()

        display(Markdown("Stage 1 Calculations"))
        self.stage_1_df()
        self.stage_1_ss()

        display(Markdown("Stage 2 Calculations"))
        self.partition_df()
        self.partition_ss()
        self.mean_squares()
        self.f_ratios()
        
        for factor in ["Factor_A", "Factor_B", "AxB"]:
            display(Markdown(f"<br>Hypothesis test for ${factor}$<br>"))

            if factor != "AxB":
                self.collapse_by_factor(factor)

            self.test_crit_values(factor)
            # self.base.write_hypotheses() - needs to be set up for all ANOVA types
            

            self.display_df(factor)
            self.base.write_decision_criteria()

            self.display_ss(factor)
            self.display_ms(factor)
            self.display_f_ratios(factor)

            # Methods from base that need to be modified
            self.base.significance = self.base.final_decision()
            self.base.write_result()
        
        


#if __name__ in "__main__":
    # FactorialData(design = (2, 3), group_n = 2).factorial_ANOVA()

