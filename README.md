# ssSIG package

**ssSIG** is an implementation of a method for enhanced discovery of statistically relevant variables using subsamples. This package extends a wide range of hypothesis tests from the [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) package to be used within the subsampling paradigm. 

## Installation

In the terminal or anaconda prompt, type:
```
pip install -i https://test.pypi.org/simple/ ssSIG
```

## Tutorial 

A tutorial is provided in the Tutorial_ssSIG.ipynb notebook. This illustrates the functioning of the ssSIG package applied to synthetic and real data (available from the `Data` directory).

## List of functions

In addition to specific options for hypothesis tests such as `alternative` which can be set to {‘two-sided’, ‘less’, ‘greater’}, the functions listed below require the following inputs which are characteristic of the subsampling method:
* `alpha`: Significance level $\alpha$ (used to calculate the empirical power for subsamples).
* `fmin`,`fmax`,`Df`: Parameters for the list of values for the subsampling fraction $f\in$ {fmin, fmin+Df, fmin+2*Df,...}.
* `Rsubsamples`: Number of random subsamples used for each value of $f$.
* `th_beta`: Linear regression for $L_{\beta}$ is restricted to values $f<$ `th_beta`.

The package also provides functions to test multiple variables for 2 and 3 samples. These functions are useful to deal with, e.g., multivariate 'omics' datasets.

Default values for the input parameters of the functions are indicated in the definition of each function.

### One sample, `x`

#### t-test
```
ttest_1samp_ssSIG(x,null_mu,Rsubsamples,fmin,fmax,Df, alternative='two-sided',alpha=0.05,th_beta=0.5)
```

### Paired samples, `x0` and `x1`

#### t-test
```
ttest_rel_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,axis=0, alternative='two-sided',alpha=0.05,th_beta=0.5)
```

#### Wilcoxon signed-rank test
```
wilcoxon_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df, alternative='two-sided', alpha=0.05,th_beta=0.5)
```

### Two independent samples

#### t-test
```
ttest_ind_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df, alternative='two-sided',alpha=0.05,th_beta=0.5)
```

#### Mann-Whitney U rank test
```
mannwhitneyu_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,alternative='two-sided',alpha=0.05,th_beta=0.5)
```

#### Kolmogorov-Smirnov test
```
ks_2samp_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,alternative='two-sided',alpha=0.05,th_beta=0.5)
```

#### Test for the logistic regression $\beta_1$ coefficient
```
logreg_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,alpha=0.05,th_beta=0.5)
```

### $k$ independent samples ($k \leq 10$), $\{x1, x2, ...\}$ 

The functions for $k$ independent samples require the data to be **input** as `data = [x1,x2,...]`, where xi (i=1,2,...) are lists containing the data for each sample.

#### One-way ANOVA
```
f_oneway_ssSIG(data,Rsubsamples,fmin,fmax,Df,alpha=0.05,th_beta=0.5)
```

#### Kruskal-Wallis H-test
```
kruskal_ssSIG(data,Rsubsamples,fmin,fmax,Df,alpha=0.05,th_beta=0.5)
```

### Multiple testing

This assumes that tests will be applied to test $n_v$ variables described by $n$ observations each. The tutorial Tutorial_ssSIG.ipynb illustrates the use of these functions to discover important metabolites.

For these functions, information on the data requires three inputs:
* `features`: Is a [pandas](https://pandas.pydata.org/) dataframe with $n_v$ columns with $n$ observations for each of the variables to be analysed and a column with the class label for each observation (labelled as {0,1} for two classes or {0,1,2} for three classes).
* `featurenames`: The list of variables to be analysed. These can be all the variables in the `features` dataframe or a subset of them.
* `classname`: Title of the column with class labels in the `features` dataframe.
    
#### Two classes
```
Multiple_tests_2classes(features,featurenames,classname,Rsubsamples,fmin,fmax,Df,SigMethod,alpha=0.05,th_beta=0.5)
```
Here, `SigMethod` can be set to 'TT' (t-test), 'MW' (Mann-Whitney), 'KS' (Kolmorov-Smirnov) and 'LR' (test for the logistic regression coefficient).

 
#### Three classes
```
Multiple_tests_3classes(features,featurenames,classname,Rsubsamples,fmin,fmax,Df,SigMethod,alpha=0.05,th_beta=0.5)
```
Here, `SigMethod` can be set to 'ANOVA' (one-way ANOVA) or 'KW' (Kruskal-Wallis H-test).
