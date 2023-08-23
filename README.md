# Overview

**ssSIG** is an implementation of a method for enhanced discovery of statistically relevant variables using subsamples. This package extends a wide range of hypothesis tests from the [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) package to be used within the subsampling paradigm. 

In addition, the package provides functions to test multiple variables for 2 and 3 samples. The Tutorial_ssSIG.ipynb notebook illustrates the 

## One sample `x`

### t-test

ttest_1samp_ssSIG(x,null_mu,Rsubsamples,fmin,fmax,Df, alternative='two-sided',alpha=0.05,th_beta=0.5)

## Paired samples `x0` and `x1`

### t-test
ttest_rel_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,axis=0, alternative='two-sided',alpha=0.05,th_beta=0.5)

### Wilcoxon signed-rank test

wilcoxon_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df, alternative='two-sided', alpha=0.05,th_beta=0.5)

## Two independent samples

### t-test
ttest_ind_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df, alternative='two-sided',alpha=0.05,th_beta=0.5)

### Mann-Whitney U rank test
mannwhitneyu_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,alternative='two-sided',alpha=0.05,th_beta=0.5)

### Kolmogorov-Smirnov test
ks_2samp_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,alternative='two-sided',alpha=0.05,th_beta=0.5)

### Test for the logistic regression $\beta_1$ coefficient

logreg_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,alpha=0.05,th_beta=0.5)

## $k$ independent samples ($k \leq 10$), $\{x1, x2, ...\}$ 

The functions for $k$ independent samples require the data to be input as data = [x1,x2,...], where xi (i=1,2,...) are lists containing the data for each sample.

### One-way ANOVA
f_oneway_ssSIG(data,Rsubsamples,fmin,fmax,Df,alpha=0.05,th_beta=0.5)

### Kruskal-Wallis H-test
kruskal_ssSIG(data,Rsubsamples,fmin,fmax,Df,alpha=0.05,th_beta=0.5)

## Multiple testing

* Multiple_tests_2classes(features,featurenames,classname,nrealisations,fmin,fmax,Df,SigMethod,alpha=0.05,th_beta=0.5)
* Multiple_tests_3classes(features,featurenames,classname,nrealisations,fmin,fmax,Df,SigMethod,alpha=0.05,th_beta=0.5)
