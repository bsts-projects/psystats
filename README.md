# Sample Problems for an Undergraduate Statistics in Psychology Course

An ongoing project to create a series of juypter notebooks that can be used to randomly generate sample problems and their worked solutions using python

This readme file needs significant updates

## Major update
Starting with hypothesis testing, individual juypter notebooks call the stats_project.py file.  The bulk of the code was written when I had *very little* experience with classes in python and much of this should be restructured.  Currently, the code is being adapted to use quarto to render accessible .html documents to meet tile II accessibility guidelines. 

## Documents to produce accessible .html practice problems
Each document will produces 5 practice problems with a n
* z-test_doc: *hypothesis testing with z-scores*
* os-t-test_doc: *one-sample t-tests*
* is-t-test_doc: *independent-samples t-tests*
* ds-t-test_doc: *dependent-samples t-tests*


When rendering the documents the yaml header option `embed-resources:` should be set to `true` to embedded math libraries into the file which seems to be necessary for correctly displaying the document when uploaded directly to Brightspace.

## Title II Updates in the immediate queue
* One-Way ANOVA
* Repeated-Measures ANOVA

## Title II Updates remaining
* standard deviation exercises
* basic z-scores and using z-scores with sample means

## Functionality to add
* correlation
* regression

