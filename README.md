---
title: Sample Problem Generator for an Undergraduate Statistics in Psychology Course
author: Guenther, Benjamin A.
project:
  type: website
  output-dir: docs
format: 
  html: default
---

This an ongoing project to create a series of ~~juypter notebooks~~ .qmd files that can be used to randomly generate sample problems and their worked solutions using python

This readme file needs significant updates

# Major update
Starting with hypothesis testing, individual .qmd files call the stats_project.py file.  The bulk of the code was written when I had *very little* experience with classes in python and much of this should be restructured.  Currently, the code is being adapted to use quarto to render accessible .html documents to meet tile II accessibility guidelines. 

# Documents to produce accessible .html practice problems
Each document will produces 30 practice problems in a .html slideshow rendered with revealjs.  Each slide is scrollable and contains a single problem.  Problems are generated with a randomly determined sample size between 5 and 15 per group.  When more than one group, sample size is equivalent between groups.<br>

* 08 - Hypothesis_Test_z-Scores.qmd: *hypothesis testing with z-scores*
* 09 - Single-Sample_t-Tests.qmd: *one-sample t-tests*
* 10 - Independent-Samples_t-Tests.qmd: *independent-samples t-tests*
* 11 - Dependent-Samples_t-Tests.qmd: *dependent-samples t-tests*
* 12 - one-way_ANOVA.qmd: *one-way ANOVA*
* 13 - repeated-measures_ANOVA: *repeated-measures ANOVA*
* 14 - factorial_ANOVA: *two-factor ANOVA*

# Title II Updates in the immediate queue
* ~~One-Way ANOVA~~
* ~~Repeated-Measures ANOVA~~

# Title II Updates remaining
* standard deviation exercises
* basic z-scores and using z-scores with sample means

# Functionality to add
* correlation
* regression
* ~~two-factor ANOVA~~
* Post-hoc testing for ANOVA
* Alt-text for figures.  Can the alt- and descriptive-text be auto generated based on the matplotlib input?
* ~~include specifying the hypotheses for NHST (null and alternative)~~

# Future directions
* seperate defining question from displaying the results
    * consider adding: `class Question:` that will hold the question (this might not be necessary)
    * consider adding: `class Assignment:` to hold a collection of question objects.  
        * will output a formatted assignment and it's answer key
* add functionality to specify some details when creating the question such as the distribution
    * add common standardized distributions as options
* expand the text of the question, inlcude a list of common variables
* switch from slide-based format to a book
    * combine all into a single book style document.
    * add instructions and a full walkthrough for each problem type