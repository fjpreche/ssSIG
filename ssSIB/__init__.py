# -*- coding: utf-8 -*-
"""
Script with functions to discover statistically significant differences in small datasets

Created on Sat Mar 25 18:52:13 2023

@author: s05fp2
"""

#----------------------------------------------------------------
#----------------------------------------------------------------
# Python functions
#----------------------------------------------------------------
#----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hypothesis tests (https://docs.scipy.org/doc/scipy/reference/stats.html)
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy.stats import f_oneway
from scipy.stats import kruskal

import scipy.stats as ss

import statsmodels.api as smapi
from sklearn.preprocessing import StandardScaler

   
#----------------------------------------------------------------
#----------------------------------------------------------------
# Statistical functions
#----------------------------------------------------------------
#----------------------------------------------------------------
def entropy(f):
    if f == 0 or f == 1:
        s = 0
    else:
        s = -(f*np.log(f)+(1-f)*np.log(1-f))
    return s
    
#----------------------------------------------------------------
#----------------------------------------------------------------
# Test functions - One sample
#----------------------------------------------------------------
#----------------------------------------------------------------

def ttest_1samp_ssSIG(data,null_mu,Rsubsamples,fmin,fmax,Df,nan_policy='propagate', alternative='two-sided',alpha=0.05,th_beta=0.5):
    n0 = len(data)
    imax = int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]
    mean_logpval = np.zeros(imax+1)
    median_logpval = np.zeros(imax+1)
    sd_logpval = np.zeros(imax+1)
    logpv_qL = np.zeros(imax+1)
    logpv_qU = np.zeros(imax+1)
    power = np.zeros(imax+1)
    
    for i in range(imax+1):
        f = flist[i]

        nr = Rsubsamples
        ln_nent = Rsubsamples*entropy(f)

        if np.log(Rsubsamples)>ln_nent:
            nr = np.floor(ln_nent).astype(int)
        nr = np.max([nr,1])
        #print('nr = '+str(nr))
        
        pr = np.zeros(nr)
        power_r = 0
        for j in range(nr):
            n0temp = int(np.round(f*n0))

            a = np.random.choice(data,n0temp,replace=False)

            s,p = ttest_1samp(a,null_mu,nan_policy=nan_policy,alternative=alternative)

            pr[j] = p #p*len(met_names)   
            if pr[j] <= alpha:
                power_r = power_r + 1

        mean_logpval[i] = np.mean(np.log(pr))
        median_logpval[i] = np.median(np.log(pr))
        sd_logpval[i] = np.std(np.log(pr))
        logpv_qL[i] = np.quantile(np.log(pr),0.025)
        logpv_qU[i] = np.quantile(np.log(pr),1-0.025)
        power[i] = power_r/nr

    #----
    # --- Linear fit to ln(P) vs f
    #----
    X = smapi.add_constant(flist)
    y = mean_logpval

    model = smapi.OLS(y,X)
    results = model.fit()

    slope_Lnpval = results.params[1]

    ciL_slope_Lnpval = results.conf_int()[1][0]
    ciU_slope_Lnpval = results.conf_int()[1][1]

    intercept_Lnpval = results.params[0]

    ciL_intercept_Lnpval = results.conf_int()[0][0]
    ciU_intercept_Lnpval = results.conf_int()[0][1]

    R2 = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb = ss.t.cdf(results.tvalues[1], df)
    
    #----
    #------ Linear fit to ln(fnr) vs f. fnr = 1-power is the false negative rate
    #----
    fnr = 1-power
    
    X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
    y = np.log(fnr[np.array(flist)<th_beta])

    model = smapi.OLS(y,X,missing='drop')
    results = model.fit()

    slope_Lnpval_fnr = results.params[1]

    ciL_slope_Lnpval_fnr = results.conf_int()[1][0]
    ciU_slope_Lnpval_fnr = results.conf_int()[1][1]

    intercept_Lnpval_fnr = results.params[0]

    ciL_intercept_Lnpval_fnr = results.conf_int()[0][0]
    ciU_intercept_Lnpval_fnr = results.conf_int()[0][1]

    R2_fnr = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb_fnr = ss.t.cdf(results.tvalues[1], df)

    return pvalb,R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,flist,mean_logpval,median_logpval,sd_logpval,logpv_qL,logpv_qU,power,pvalb_fnr,R2_fnr,slope_Lnpval_fnr,ciL_slope_Lnpval_fnr,ciU_slope_Lnpval_fnr,intercept_Lnpval_fnr,ciL_intercept_Lnpval_fnr,ciU_intercept_Lnpval_fnr

#----------------------------------------------------------------
#----------------------------------------------------------------
# Test functions - One sample, two measures or paired replicaties
#----------------------------------------------------------------
#----------------------------------------------------------------

## ----------------------------------------------------------------------------
## t-test on TWO RELATED samples of scores, a and b
## ----------------------------------------------------------------------------

def ttest_rel_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,axis=0, alternative='two-sided', keepdims=False,alpha=0.05,th_beta=0.5):
    n0 = len(x0)
    n1 = len(x1)
    imax = int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]
    mean_logpval = np.zeros(imax+1)
    median_logpval = np.zeros(imax+1)
    sd_logpval = np.zeros(imax+1)
    logpv_qL = np.zeros(imax+1)
    logpv_qU = np.zeros(imax+1)
    power = np.zeros(imax+1)
    
    for i in range(imax+1):
        f = flist[i]

        nr = Rsubsamples
        ln_nent = Rsubsamples*entropy(f)

        if np.log(Rsubsamples)>ln_nent:
            nr = np.floor(ln_nent).astype(int)
        nr = np.max([nr,1])

        pr = np.zeros(nr)
        power_r = 0
        for j in range(nr):
            n0temp = int(np.round(f*n0))
            n1temp = int(np.round(f*n1))
            
            x0r = np.random.choice(x0,n0temp,replace=False)
            x1r = np.random.choice(x1,n1temp,replace=False)

            s,p = ttest_rel(x0r,x1r, axis=0, alternative=alternative)

            pr[j] = p #p*len(met_names)
            if pr[j] <= alpha:
                power_r = power_r + 1
            

        mean_logpval[i] = np.mean(np.log(pr))
        median_logpval[i] = np.median(np.log(pr))
        sd_logpval[i] = np.std(np.log(pr))
        logpv_qL[i] = np.quantile(np.log(pr),0.025)
        logpv_qU[i] = np.quantile(np.log(pr),1-0.025)
        power[i] = power_r/nr
        
    #----
    # --- Linear fit to ln(P) vs f
    #----
    X = smapi.add_constant(flist)
    y = mean_logpval

    model = smapi.OLS(y,X)
    results = model.fit()

    slope_Lnpval = results.params[1]

    ciL_slope_Lnpval = results.conf_int()[1][0]
    ciU_slope_Lnpval = results.conf_int()[1][1]

    intercept_Lnpval = results.params[0]

    ciL_intercept_Lnpval = results.conf_int()[0][0]
    ciU_intercept_Lnpval = results.conf_int()[0][1]

    R2 = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb = ss.t.cdf(results.tvalues[1], df)
    
    #----
    #------ Linear fit to ln(fnr) vs f. fnr = 1-power is the false negative rate
    #----
    fnr = 1-power
    
    X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
    y = np.log(fnr[np.array(flist)<th_beta])

    model = smapi.OLS(y,X,missing='drop')
    results = model.fit()

    slope_Lnpval_fnr = results.params[1]

    ciL_slope_Lnpval_fnr = results.conf_int()[1][0]
    ciU_slope_Lnpval_fnr = results.conf_int()[1][1]

    intercept_Lnpval_fnr = results.params[0]

    ciL_intercept_Lnpval_fnr = results.conf_int()[0][0]
    ciU_intercept_Lnpval_fnr = results.conf_int()[0][1]

    R2_fnr = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb_fnr = ss.t.cdf(results.tvalues[1], df)

    return pvalb,R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,flist,mean_logpval,median_logpval,sd_logpval,logpv_qL,logpv_qU,power,pvalb_fnr,R2_fnr,slope_Lnpval_fnr,ciL_slope_Lnpval_fnr,ciU_slope_Lnpval_fnr,intercept_Lnpval_fnr,ciL_intercept_Lnpval_fnr,ciU_intercept_Lnpval_fnr


## ----------------------------------------------------------------------------
## Wilcoxon signed-rank test.
## Tests the null hypothesis that two related paired samples come from the same distribution. 
## In particular, it tests whether the distribution of the differences x0 - x1 is symmetric about zero. 
## It is a non-parametric version of the paired T-test.
## ----------------------------------------------------------------------------

def wilcoxon_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df, alternative='two-sided', alpha=0.05,th_beta=0.5):
    n0 = len(x0)
    n1 = len(x1)
    imax = int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]
    mean_logpval = np.zeros(imax+1)
    median_logpval = np.zeros(imax+1)
    sd_logpval = np.zeros(imax+1)
    logpv_qL = np.zeros(imax+1)
    logpv_qU = np.zeros(imax+1)
    power = np.zeros(imax+1)
        
    for i in range(imax+1):
        f = flist[i]

        nr = Rsubsamples
        ln_nent = Rsubsamples*entropy(f)

        if np.log(Rsubsamples)>ln_nent:
            nr = np.floor(ln_nent).astype(int)
        nr = np.max([nr,1])

        pr = np.zeros(nr)
        power_r = 0
        for j in range(nr):
            n0temp = int(np.round(f*n0))
            n1temp = int(np.round(f*n1))
            
            x0r = np.random.choice(x0,n0temp,replace=False)
            x1r = np.random.choice(x1,n1temp,replace=False)

            s,p = wilcoxon(x0r,x1r, alternative=alternative)
            #wilcoxon(x0r,x1r,zero_method=zero_method, correction=correction, alternative=alternative, method=method, axis=0, nan_policy=nan_policy)

            pr[j] = p #p*len(met_names)
            if pr[j] <= alpha:
                power_r = power_r + 1
            

        mean_logpval[i] = np.mean(np.log(pr))
        median_logpval[i] = np.median(np.log(pr))
        sd_logpval[i] = np.std(np.log(pr))
        logpv_qL[i] = np.quantile(np.log(pr),0.025)
        logpv_qU[i] = np.quantile(np.log(pr),1-0.025)
        power[i] = power_r/nr

    #----
    # --- Linear fit to ln(P) vs f
    #----
    X = smapi.add_constant(flist)
    y = mean_logpval

    model = smapi.OLS(y,X)
    results = model.fit()

    slope_Lnpval = results.params[1]

    ciL_slope_Lnpval = results.conf_int()[1][0]
    ciU_slope_Lnpval = results.conf_int()[1][1]

    intercept_Lnpval = results.params[0]

    ciL_intercept_Lnpval = results.conf_int()[0][0]
    ciU_intercept_Lnpval = results.conf_int()[0][1]

    R2 = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb = ss.t.cdf(results.tvalues[1], df)
    
    #----
    #------ Linear fit to ln(fnr) vs f. fnr = 1-power is the false negative rate
    #----
    fnr = 1-power
    
    X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
    y = np.log(fnr[np.array(flist)<th_beta])

    model = smapi.OLS(y,X,missing='drop')
    results = model.fit()

    slope_Lnpval_fnr = results.params[1]

    ciL_slope_Lnpval_fnr = results.conf_int()[1][0]
    ciU_slope_Lnpval_fnr = results.conf_int()[1][1]

    intercept_Lnpval_fnr = results.params[0]

    ciL_intercept_Lnpval_fnr = results.conf_int()[0][0]
    ciU_intercept_Lnpval_fnr = results.conf_int()[0][1]

    R2_fnr = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb_fnr = ss.t.cdf(results.tvalues[1], df)

    return pvalb,R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,flist,mean_logpval,median_logpval,sd_logpval,logpv_qL,logpv_qU,power,pvalb_fnr,R2_fnr,slope_Lnpval_fnr,ciL_slope_Lnpval_fnr,ciU_slope_Lnpval_fnr,intercept_Lnpval_fnr,ciL_intercept_Lnpval_fnr,ciU_intercept_Lnpval_fnr


#----------------------------------------------------------------
#----------------------------------------------------------------
# Test functions - Two independent samples
#----------------------------------------------------------------
#----------------------------------------------------------------

## ----------------------------------------------------------------------------
## t-test for the means of two independent samples of scores.
## ----------------------------------------------------------------------------

def ttest_ind_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df, alternative='two-sided',permutations=None, random_state=None,alpha=0.05,th_beta=0.5):
    n0 = len(x0)
    n1 = len(x1)
    imax = int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]
    mean_logpval = np.zeros(imax+1)
    median_logpval = np.zeros(imax+1)
    sd_logpval = np.zeros(imax+1)
    logpv_qL = np.zeros(imax+1)
    logpv_qU = np.zeros(imax+1)
    power = np.zeros(imax+1)
    
    for i in range(imax+1):
        f = flist[i]

        nr = Rsubsamples

#        ln_nent = Rsubsamples*entropy(f)
#        if np.log(Rsubsamples)>ln_nent:
#            nr = np.floor(ln_nent).astype(int)
#        nr = np.max([nr,1])

        pr = np.zeros(nr)
        power_r = 0
        for j in range(nr):
            n0temp = int(np.round(f*n0))
            n1temp = int(np.round(f*n1))
            
            x0r = np.random.choice(x0,n0temp,replace=False)
            x1r = np.random.choice(x1,n1temp,replace=False)

            #s,p = ttest_ind(x0r,x1r, alternative=alternative,permutations=permutations, random_state=random_state)
            s,p = ttest_ind(x0r,x1r,alternative=alternative)
            
            pr[j] = p #p*len(met_names)   
            if pr[j] <= alpha:
                power_r = power_r + 1

        mean_logpval[i] = np.mean(np.log(pr))
        median_logpval[i] = np.median(np.log(pr))
        sd_logpval[i] = np.std(np.log(pr))
        logpv_qL[i] = np.quantile(np.log(pr),0.025)
        logpv_qU[i] = np.quantile(np.log(pr),1-0.025)
        power[i] = power_r/nr
        
    #----
    # --- Linear fit to ln(P) vs f
    #----
    X = smapi.add_constant(flist)
    y = mean_logpval

    model = smapi.OLS(y,X)
    results = model.fit()

    slope_Lnpval = results.params[1]

    ciL_slope_Lnpval = results.conf_int()[1][0]
    ciU_slope_Lnpval = results.conf_int()[1][1]

    intercept_Lnpval = results.params[0]

    ciL_intercept_Lnpval = results.conf_int()[0][0]
    ciU_intercept_Lnpval = results.conf_int()[0][1]

    R2 = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb = ss.t.cdf(results.tvalues[1], df)
    
    #----
    #------ Linear fit to ln(fnr) vs f. fnr = 1-power is the false negative rate
    #----
    fnr = 1-power
    
    X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
    y = np.log(fnr[np.array(flist)<th_beta])

    model = smapi.OLS(y,X,missing='drop')
    results = model.fit()

    slope_Lnpval_fnr = results.params[1]

    ciL_slope_Lnpval_fnr = results.conf_int()[1][0]
    ciU_slope_Lnpval_fnr = results.conf_int()[1][1]

    intercept_Lnpval_fnr = results.params[0]

    ciL_intercept_Lnpval_fnr = results.conf_int()[0][0]
    ciU_intercept_Lnpval_fnr = results.conf_int()[0][1]

    R2_fnr = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb_fnr = ss.t.cdf(results.tvalues[1], df)

    return pvalb,R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,flist,mean_logpval,median_logpval,sd_logpval,logpv_qL,logpv_qU,power,pvalb_fnr,R2_fnr,slope_Lnpval_fnr,ciL_slope_Lnpval_fnr,ciU_slope_Lnpval_fnr,intercept_Lnpval_fnr,ciL_intercept_Lnpval_fnr,ciU_intercept_Lnpval_fnr


## ----------------------------------------------------------------------------
## Logistic regression for two independent samples of scores.
## ----------------------------------------------------------------------------

def logreg_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,alpha=0.05,th_beta=0.5): #,nan_policy='propagate', alternative='two-sided',permutations=None, random_state=None):
    n0 = len(x0)
    n1 = len(x1)
    imax = int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]
    mean_logpval = np.zeros(imax+1)
    median_logpval = np.zeros(imax+1)
    sd_logpval = np.zeros(imax+1)
    logpv_qL = np.zeros(imax+1)
    logpv_qU = np.zeros(imax+1)
    power = np.zeros(imax+1)
    
    for i in range(imax+1):
        f = flist[i]

        nr = Rsubsamples
        ln_nent = Rsubsamples*entropy(f)

        if np.log(Rsubsamples)>ln_nent:
            nr = np.floor(ln_nent).astype(int)
        nr = np.max([nr,1])

        pr = np.zeros(nr)
        power_r = 0
        for j in range(nr):
            n0temp = int(np.round(f*n0))
            n1temp = int(np.round(f*n1))
            
            x0r = np.random.choice(x0,n0temp,replace=False)
            x1r = np.random.choice(x1,n1temp,replace=False)
            
            x = np.concatenate([x0r,x1r])
            y = np.concatenate([np.zeros(len(x0r)),np.zeros(len(x1r))+1])

            # Scaling
            x = x.reshape(-1,1)
#
            scaler = StandardScaler().fit(x)
            x_scaled = scaler.transform(x)
            xs = pd.DataFrame(data=x_scaled)
#            xs = x.reshape(-1,1)
            
            y = y.reshape(-1,1)

            log_reg = smapi.Logit(y,xs).fit(disp=0)

            p = log_reg.pvalues.values    

            pr[j] = p #p*len(met_names)   
            if pr[j] <= alpha:
                power_r = power_r + 1

        mean_logpval[i] = np.mean(np.log(pr))
        median_logpval[i] = np.median(np.log(pr))
        sd_logpval[i] = np.std(np.log(pr))
        logpv_qL[i] = np.quantile(np.log(pr),0.025)
        logpv_qU[i] = np.quantile(np.log(pr),1-0.025)
        power[i] = power_r/nr

    #----
    # --- Linear fit to ln(P) vs f
    #----
    X = smapi.add_constant(flist)
    y = mean_logpval

    model = smapi.OLS(y,X)
    results = model.fit()

    slope_Lnpval = results.params[1]

    ciL_slope_Lnpval = results.conf_int()[1][0]
    ciU_slope_Lnpval = results.conf_int()[1][1]

    intercept_Lnpval = results.params[0]

    ciL_intercept_Lnpval = results.conf_int()[0][0]
    ciU_intercept_Lnpval = results.conf_int()[0][1]

    R2 = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb = ss.t.cdf(results.tvalues[1], df)
    
    #----
    #------ Linear fit to ln(fnr) vs f. fnr = 1-power is the false negative rate
    #----
    fnr = 1-power
    
    X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
    y = np.log(fnr[np.array(flist)<th_beta])

    model = smapi.OLS(y,X,missing='drop')
    results = model.fit()

    slope_Lnpval_fnr = results.params[1]

    ciL_slope_Lnpval_fnr = results.conf_int()[1][0]
    ciU_slope_Lnpval_fnr = results.conf_int()[1][1]

    intercept_Lnpval_fnr = results.params[0]

    ciL_intercept_Lnpval_fnr = results.conf_int()[0][0]
    ciU_intercept_Lnpval_fnr = results.conf_int()[0][1]

    R2_fnr = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb_fnr = ss.t.cdf(results.tvalues[1], df)

    return pvalb,R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,flist,mean_logpval,median_logpval,sd_logpval,logpv_qL,logpv_qU,power,pvalb_fnr,R2_fnr,slope_Lnpval_fnr,ciL_slope_Lnpval_fnr,ciU_slope_Lnpval_fnr,intercept_Lnpval_fnr,ciL_intercept_Lnpval_fnr,ciU_intercept_Lnpval_fnr

## ----------------------------------------------------------------------------
## Mann-Whitney U rank test on two independent samples
## ----------------------------------------------------------------------------

def mannwhitneyu_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,alternative='two-sided',alpha=0.05,th_beta=0.5):
    n0 = len(x0)
    n1 = len(x1)
    imax = int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]
    mean_logpval = np.zeros(imax+1)
    median_logpval = np.zeros(imax+1)
    sd_logpval = np.zeros(imax+1)
    logpv_qL = np.zeros(imax+1)
    logpv_qU = np.zeros(imax+1)
    power = np.zeros(imax+1)    
    
    for i in range(imax+1):
        f = flist[i]

        nr = Rsubsamples
        ln_nent = Rsubsamples*entropy(f)

        if np.log(Rsubsamples)>ln_nent:
            nr = np.floor(ln_nent).astype(int)
        nr = np.max([nr,1])

        pr = np.zeros(nr)
        power_r = 0
        for j in range(nr):
            n0temp = int(np.round(f*n0))
            n1temp = int(np.round(f*n1))
            
            x0r = np.random.choice(x0,n0temp,replace=False)
            x1r = np.random.choice(x1,n1temp,replace=False)

            s,p = mannwhitneyu(x0r,x1r,alternative=alternative)

            pr[j] = p #p*len(met_names)   
            if pr[j] <= alpha:
                power_r = power_r + 1

        mean_logpval[i] = np.mean(np.log(pr))
        median_logpval[i] = np.median(np.log(pr))
        sd_logpval[i] = np.std(np.log(pr))
        logpv_qL[i] = np.quantile(np.log(pr),0.025)
        logpv_qU[i] = np.quantile(np.log(pr),1-0.025)
        power[i] = power_r/nr
        
    #----
    # --- Linear fit to ln(P) vs f
    #----
    X = smapi.add_constant(flist)
    y = mean_logpval

    model = smapi.OLS(y,X)
    results = model.fit()

    slope_Lnpval = results.params[1]

    ciL_slope_Lnpval = results.conf_int()[1][0]
    ciU_slope_Lnpval = results.conf_int()[1][1]

    intercept_Lnpval = results.params[0]

    ciL_intercept_Lnpval = results.conf_int()[0][0]
    ciU_intercept_Lnpval = results.conf_int()[0][1]

    R2 = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb = ss.t.cdf(results.tvalues[1], df)
    
    #----
    #------ Linear fit to ln(fnr) vs f. fnr = 1-power is the false negative rate
    #----
    fnr = 1-power
    
    X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
    y = np.log(fnr[np.array(flist)<th_beta])

    model = smapi.OLS(y,X,missing='drop')
    results = model.fit()

    slope_Lnpval_fnr = results.params[1]

    ciL_slope_Lnpval_fnr = results.conf_int()[1][0]
    ciU_slope_Lnpval_fnr = results.conf_int()[1][1]

    intercept_Lnpval_fnr = results.params[0]

    ciL_intercept_Lnpval_fnr = results.conf_int()[0][0]
    ciU_intercept_Lnpval_fnr = results.conf_int()[0][1]

    R2_fnr = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb_fnr = ss.t.cdf(results.tvalues[1], df)

    return pvalb,R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,flist,mean_logpval,median_logpval,sd_logpval,logpv_qL,logpv_qU,power,pvalb_fnr,R2_fnr,slope_Lnpval_fnr,ciL_slope_Lnpval_fnr,ciU_slope_Lnpval_fnr,intercept_Lnpval_fnr,ciL_intercept_Lnpval_fnr,ciU_intercept_Lnpval_fnr


## ----------------------------------------------------------------------------
## Two-sample Kolmogorov-Smirnov test for goodness of fit.
## This test compares the underlying continuous distributions F(x) and G(x) of two independent samples.
## ----------------------------------------------------------------------------

def ks_2samp_ssSIG(x0,x1,Rsubsamples,fmin,fmax,Df,alternative='two-sided',alpha=0.05,th_beta=0.5):
    n0 = len(x0)
    n1 = len(x1)
    imax = int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]
    mean_logpval = np.zeros(imax+1)
    median_logpval = np.zeros(imax+1)
    sd_logpval = np.zeros(imax+1)
    logpv_qL = np.zeros(imax+1)
    logpv_qU = np.zeros(imax+1)
    power = np.zeros(imax+1)
    
    for i in range(imax+1):
        f = flist[i]

        nr = Rsubsamples
        ln_nent = Rsubsamples*entropy(f)

        if np.log(Rsubsamples)>ln_nent:
            nr = np.floor(ln_nent).astype(int)
        nr = np.max([nr,1])

        pr = np.zeros(nr)
        power_r = 0
        for j in range(nr):
            n0temp = int(np.round(f*n0))
            n1temp = int(np.round(f*n1))
            
            x0r = np.random.choice(x0,n0temp,replace=False)
            x1r = np.random.choice(x1,n1temp,replace=False)

            s,p = ks_2samp(x0r,x1r,alternative=alternative)

            pr[j] = p #p*len(met_names)   
            if pr[j] <= alpha:
                power_r = power_r + 1

        mean_logpval[i] = np.mean(np.log(pr))
        median_logpval[i] = np.median(np.log(pr))
        sd_logpval[i] = np.std(np.log(pr))
        logpv_qL[i] = np.quantile(np.log(pr),0.025)
        logpv_qU[i] = np.quantile(np.log(pr),1-0.025)
        power[i] = power_r/nr
        
    #----
    # --- Linear fit to ln(P) vs f
    #----
    X = smapi.add_constant(flist)
    y = mean_logpval

    model = smapi.OLS(y,X)
    results = model.fit()

    slope_Lnpval = results.params[1]

    ciL_slope_Lnpval = results.conf_int()[1][0]
    ciU_slope_Lnpval = results.conf_int()[1][1]

    intercept_Lnpval = results.params[0]

    ciL_intercept_Lnpval = results.conf_int()[0][0]
    ciU_intercept_Lnpval = results.conf_int()[0][1]

    R2 = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb = ss.t.cdf(results.tvalues[1], df)
    
    #----
    #------ Linear fit to ln(fnr) vs f. fnr = 1-power is the false negative rate
    #----
    fnr = 1-power
    
    X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
    y = np.log(fnr[np.array(flist)<th_beta])

    model = smapi.OLS(y,X,missing='drop')
    results = model.fit()

    slope_Lnpval_fnr = results.params[1]

    ciL_slope_Lnpval_fnr = results.conf_int()[1][0]
    ciU_slope_Lnpval_fnr = results.conf_int()[1][1]

    intercept_Lnpval_fnr = results.params[0]

    ciL_intercept_Lnpval_fnr = results.conf_int()[0][0]
    ciU_intercept_Lnpval_fnr = results.conf_int()[0][1]

    R2_fnr = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb_fnr = ss.t.cdf(results.tvalues[1], df)

    return pvalb,R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,flist,mean_logpval,median_logpval,sd_logpval,logpv_qL,logpv_qU,power,pvalb_fnr,R2_fnr,slope_Lnpval_fnr,ciL_slope_Lnpval_fnr,ciU_slope_Lnpval_fnr,intercept_Lnpval_fnr,ciL_intercept_Lnpval_fnr,ciU_intercept_Lnpval_fnr


#----------------------------------------------------------------
#----------------------------------------------------------------
# Test functions - k independent samples
#----------------------------------------------------------------
#----------------------------------------------------------------

##-----------------------------------------------------------------------------------
# The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. 
# The test is applied to samples from two or more groups, possibly with differing sizes.
##-----------------------------------------------------------------------------------

def f_oneway_ssSIG(data,Rsubsamples,fmin,fmax,Df,alpha=0.05,th_beta=0.5): #,nan_policy='propagate', alternative='two-sided',permutations=None, random_state=None):
    k = len(data)

    nlist = []
    for i in range(k):
        nlist.append(len(data[i]))

    imax = int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]
    mean_logpval = np.zeros(imax+1)
    median_logpval = np.zeros(imax+1)
    sd_logpval = np.zeros(imax+1)
    logpv_qL = np.zeros(imax+1)
    logpv_qU = np.zeros(imax+1)
    power = np.zeros(imax+1)
    
    for i in range(imax+1):
        f = flist[i]

        nr = Rsubsamples
        ln_nent = Rsubsamples*entropy(f)

        if np.log(Rsubsamples)>ln_nent:
            nr = np.floor(ln_nent).astype(int)
        nr = np.max([nr,1])

        pr = np.zeros(nr)
        power_r = 0
        for j in range(nr):

            xr = []
            for l in range(k):
                ntemp = int(np.round(f*nlist[l]))
                xr.append(np.random.choice(data[l],ntemp,replace=False))

            if k == 2:
                s,p = f_oneway(xr[0],xr[1]) 
            if k == 3:
                s,p = f_oneway(xr[0],xr[1],xr[2]) 
            if k == 4:
                s,p = f_oneway(xr[0],xr[1],xr[2],xr[3]) 
            if k == 5:
                s,p = f_oneway(xr[0],xr[1],xr[2],xr[3],xr[4]) 
            if k == 6:
                s,p = f_oneway(xr[0],xr[1],xr[2],xr[3],xr[4],xr[5])
            if k == 7:
                s,p = f_oneway(xr[0],xr[1],xr[2],xr[3],xr[4],xr[5],xr[6])
            if k == 8:
                s,p = f_oneway(xr[0],xr[1],xr[2],xr[3],xr[4],xr[5],xr[6],xr[7])
            if k == 9:
                s,p = f_oneway(xr[0],xr[1],xr[2],xr[3],xr[4],xr[5],xr[6],xr[7],xr[8])
            if k == 10:
                s,p = f_oneway(xr[0],xr[1],xr[2],xr[3],xr[4],xr[5],xr[6],xr[7],xr[8],xr[9])

            pr[j] = p #p*len(met_names)   
            if pr[j] <= alpha:
                power_r = power_r + 1

        mean_logpval[i] = np.mean(np.log(pr))
        median_logpval[i] = np.median(np.log(pr))
        sd_logpval[i] = np.std(np.log(pr))
        logpv_qL[i] = np.quantile(np.log(pr),0.025)
        logpv_qU[i] = np.quantile(np.log(pr),1-0.025)
        power[i] = power_r/nr


    #----
    # --- Linear fit to ln(P) vs f
    #----
    X = smapi.add_constant(flist)
    y = mean_logpval

    model = smapi.OLS(y,X)
    results = model.fit()

    slope_Lnpval = results.params[1]

    ciL_slope_Lnpval = results.conf_int()[1][0]
    ciU_slope_Lnpval = results.conf_int()[1][1]

    intercept_Lnpval = results.params[0]

    ciL_intercept_Lnpval = results.conf_int()[0][0]
    ciU_intercept_Lnpval = results.conf_int()[0][1]

    R2 = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb = ss.t.cdf(results.tvalues[1], df)
    
    #----
    #------ Linear fit to ln(fnr) vs f. fnr = 1-power is the false negative rate
    #----
    fnr = 1-power
    
    X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
    y = np.log(fnr[np.array(flist)<th_beta])

    model = smapi.OLS(y,X,missing='drop')
    results = model.fit()

    slope_Lnpval_fnr = results.params[1]

    ciL_slope_Lnpval_fnr = results.conf_int()[1][0]
    ciU_slope_Lnpval_fnr = results.conf_int()[1][1]

    intercept_Lnpval_fnr = results.params[0]

    ciL_intercept_Lnpval_fnr = results.conf_int()[0][0]
    ciU_intercept_Lnpval_fnr = results.conf_int()[0][1]

    R2_fnr = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb_fnr = ss.t.cdf(results.tvalues[1], df)

    return pvalb,R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,flist,mean_logpval,median_logpval,sd_logpval,logpv_qL,logpv_qU,power,pvalb_fnr,R2_fnr,slope_Lnpval_fnr,ciL_slope_Lnpval_fnr,ciU_slope_Lnpval_fnr,intercept_Lnpval_fnr,ciL_intercept_Lnpval_fnr,ciU_intercept_Lnpval_fnr


## ----------------------------------------------------------------------------
## The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal. 
#It is a non-parametric version of ANOVA. 
#The test works on 2 or more independent samples, which may have different sizes. 
#Note that rejecting the null hypothesis does not indicate which of the groups differs. 
#Post hoc comparisons between groups are required to determine which groups are different.
## ----------------------------------------------------------------------------

def kruskal_ssSIG(data,Rsubsamples,fmin,fmax,Df,alpha=0.05,th_beta=0.5): #,nan_policy='propagate', alternative='two-sided',permutations=None, random_state=None):
    k = len(data)

    nlist = []
    for i in range(k):
        nlist.append(len(data[i]))

    imax = int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]
    mean_logpval = np.zeros(imax+1)
    median_logpval = np.zeros(imax+1)
    sd_logpval = np.zeros(imax+1)
    logpv_qL = np.zeros(imax+1)
    logpv_qU = np.zeros(imax+1)
    power = np.zeros(imax+1)

    for i in range(imax+1):
        f = flist[i]

        nr = Rsubsamples
        ln_nent = Rsubsamples*entropy(f)

        if np.log(Rsubsamples)>ln_nent:
            nr = np.floor(ln_nent).astype(int)
        nr = np.max([nr,1])

        pr = np.zeros(nr)
        power_r = 0
        for j in range(nr):

            xr = []
            for l in range(k):
                ntemp = int(np.round(f*nlist[l]))
                xr.append(np.random.choice(data[l],ntemp,replace=False))

            if k == 2:
                s,p = kruskal(xr[0],xr[1]) 
            if k == 3:
                s,p = kruskal(xr[0],xr[1],xr[2]) 
            if k == 4:
                s,p = kruskal(xr[0],xr[1],xr[2],xr[3]) 
            if k == 5:
                s,p = kruskal(xr[0],xr[1],xr[2],xr[3],xr[4]) 
            if k == 6:
                s,p = kruskal(xr[0],xr[1],xr[2],xr[3],xr[4],xr[5])
            if k == 7:
                s,p = kruskal(xr[0],xr[1],xr[2],xr[3],xr[4],xr[5],xr[6])
            if k == 8:
                s,p = kruskal(xr[0],xr[1],xr[2],xr[3],xr[4],xr[5],xr[6],xr[7])
            if k == 9:
                s,p = kruskal(xr[0],xr[1],xr[2],xr[3],xr[4],xr[5],xr[6],xr[7],xr[8])
            if k == 10:
                s,p = kruskal(xr[0],xr[1],xr[2],xr[3],xr[4],xr[5],xr[6],xr[7],xr[8],xr[9])
            
            pr[j] = p #p*len(met_names)
            if pr[j] <= alpha:
                power_r = power_r + 1
            #The p-value for the test using the assumption that H has a chi square distribution. 
            #The p-value returned is the survival function of the chi square distribution evaluated at H.


        mean_logpval[i] = np.mean(np.log(pr))
        median_logpval[i] = np.median(np.log(pr))
        sd_logpval[i] = np.std(np.log(pr))
        logpv_qL[i] = np.quantile(np.log(pr),0.025)
        logpv_qU[i] = np.quantile(np.log(pr),1-0.025)
        power[i] = power_r/nr
        
    #----
    # --- Linear fit to ln(P) vs f
    #----
    X = smapi.add_constant(flist)
    y = mean_logpval

    model = smapi.OLS(y,X)
    results = model.fit()

    slope_Lnpval = results.params[1]

    ciL_slope_Lnpval = results.conf_int()[1][0]
    ciU_slope_Lnpval = results.conf_int()[1][1]

    intercept_Lnpval = results.params[0]

    ciL_intercept_Lnpval = results.conf_int()[0][0]
    ciU_intercept_Lnpval = results.conf_int()[0][1]

    R2 = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb = ss.t.cdf(results.tvalues[1], df)
    
    #----
    #------ Linear fit to ln(fnr) vs f. fnr = 1-power is the false negative rate
    #----
    fnr = 1-power
    
    X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
    y = np.log(fnr[np.array(flist)<th_beta])

    model = smapi.OLS(y,X,missing='drop')
    results = model.fit()

    slope_Lnpval_fnr = results.params[1]

    ciL_slope_Lnpval_fnr = results.conf_int()[1][0]
    ciU_slope_Lnpval_fnr = results.conf_int()[1][1]

    intercept_Lnpval_fnr = results.params[0]

    ciL_intercept_Lnpval_fnr = results.conf_int()[0][0]
    ciU_intercept_Lnpval_fnr = results.conf_int()[0][1]

    R2_fnr = results.rsquared

    #Two-sided test
    #pvalb = results.pvalues[1]

    # One-sided p-value for the slope (negative slope is significant)
    df = results.df_resid
    pvalb_fnr = ss.t.cdf(results.tvalues[1], df)

    return pvalb,R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,flist,mean_logpval,median_logpval,sd_logpval,logpv_qL,logpv_qU,power,pvalb_fnr,R2_fnr,slope_Lnpval_fnr,ciL_slope_Lnpval_fnr,ciU_slope_Lnpval_fnr,intercept_Lnpval_fnr,ciL_intercept_Lnpval_fnr,ciU_intercept_Lnpval_fnr


#----------------------------------------------------------------
#----------------------------------------------------------------
# Multiple testing
#----------------------------------------------------------------
#----------------------------------------------------------------

#### --- Two classes (t-test, Mann-Whitney, Kolmogorov-Smirnov, Logistic regression)
def Multiple_tests_2classes(features,featurenames,classname,Rsubsamples,fmin,fmax,Df,SigMethod,alpha=0.05,th_beta=0.5):
    
    n0 = len(features[classname][features[classname] == 0])
    n1 = len(features[classname][features[classname] == 1])
    met_names = featurenames

    imax = np.int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]

    slope_Lnpval = np.zeros(len(met_names))
    ciL_slope_Lnpval = np.zeros(len(met_names))
    ciU_slope_Lnpval = np.zeros(len(met_names))

    intercept_Lnpval = np.zeros(len(met_names))
    ciL_intercept_Lnpval = np.zeros(len(met_names))
    ciU_intercept_Lnpval = np.zeros(len(met_names))

    R2 = np.zeros(len(met_names))
    pvalb = np.zeros(len(met_names))
    pvalb_beta = np.zeros(len(met_names))
    
    for m in range(len(met_names)):
    #for m in range(10):
        print('feature m ='+str(m)+' of '+str(len(met_names)))
        met = met_names[m] #sigBH0[3]

        pval = np.zeros(imax+1)
        power = np.zeros(imax+1)
        
        for i in range(imax+1):
            f = flist[i]

            nr = Rsubsamples
            ln_nent = Rsubsamples*entropy(f)
            
            if np.log(Rsubsamples)>ln_nent:
                nr = np.floor(ln_nent).astype(int)
            nr = np.max([nr,1])
            
            lnpi = np.zeros(nr)
            power_r = 0
            for j in range(nr):
                n0temp = np.int(np.round(f*n0))
                n1temp = np.int(np.round(f*n1))

                a = features[[met]][features['class'] == 0].sample(n=n0temp)
                b = features[[met]][features['class'] == 1].sample(n=n1temp)

                
                if SigMethod == "TT":
                    s,p = ttest_ind(a,b)
                if SigMethod == "MW":
                    s,p = mannwhitneyu(a,b)
                if SigMethod == "KS":
                    s,p = ks_2samp(a[met],b[met])
                if SigMethod == "LR":
                    x = np.concatenate([a,b])
                    y = np.concatenate([np.zeros(len(a)),np.zeros(len(b))+1])
        
                    # Scaling
                    x = x.reshape(-1,1)
        #
                    scaler = StandardScaler().fit(x)
                    x_scaled = scaler.transform(x)
                    xs = pd.DataFrame(data=x_scaled)
        #            xs = x.reshape(-1,1)
                    
                    y = y.reshape(-1,1)
        
                    log_reg = smapi.Logit(y,xs).fit(disp=0)
        
                    p = log_reg.pvalues.values    

                if p <= alpha:
                    power_r = power_r + 1
                
                lnpi[j] = np.log(p) #p*len(met_names)   

            pval[i] = np.mean(lnpi)
            power[i] = power_r/nr
            
        # - Fitting a linear dependence 
        X = smapi.add_constant(flist)
        y = pval

        model = smapi.OLS(y,X)
        results = model.fit()

        slope_Lnpval[m] = results.params[1]

        ciL_slope_Lnpval[m] = results.conf_int()[1][0]
        ciU_slope_Lnpval[m] = results.conf_int()[1][1]

        intercept_Lnpval[m] = results.params[0]

        ciL_intercept_Lnpval[m] = results.conf_int()[0][0]
        ciU_intercept_Lnpval[m] = results.conf_int()[0][1]

        R2[m] = results.rsquared
        #pvalb[m] = results.pvalues[1]
        
        # One-sided p-value for the slope (negative slope is significant)
        df = results.df_resid
        pvalb[m] = ss.t.cdf(results.tvalues[1], df)
        
        
        #----
        #------ Linear fit to ln(beta) vs f. beta = 1-power is the false negative rate
        #----
        beta = 1-power
        
        #print(beta)
        
        X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
        y = np.log(beta[np.array(flist)<th_beta])
    
        model = smapi.OLS(y,X,missing='drop')
        results = model.fit()
        #
        df = results.df_resid
        pvalb_beta[m] = ss.t.cdf(results.tvalues[1], df)
        
        
    return R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,pvalb,pvalb_beta


#### --- Three classes (one way ANOVA or Kruskal-Wallis)
#--
def Multiple_tests_3classes(features,featurenames,classname,Rsubsamples,fmin,fmax,Df,SigMethod,alpha=0.05,th_beta=0.5):
    
    n0 = len(features[classname][features[classname] == 0])
    n1 = len(features[classname][features[classname] == 1])
    n2 = len(features[classname][features[classname] == 2])
    met_names = featurenames

    imax = np.int(np.floor((fmax-fmin)/Df))

    flist = [fmin + i*Df for i in range(imax+1)]

    slope_Lnpval = np.zeros(len(met_names))
    ciL_slope_Lnpval = np.zeros(len(met_names))
    ciU_slope_Lnpval = np.zeros(len(met_names))

    intercept_Lnpval = np.zeros(len(met_names))
    ciL_intercept_Lnpval = np.zeros(len(met_names))
    ciU_intercept_Lnpval = np.zeros(len(met_names))

    R2 = np.zeros(len(met_names))
    pvalb = np.zeros(len(met_names))
    pvalb_beta = np.zeros(len(met_names))
    
    for m in range(len(met_names)):
    #for m in range(10):
        print('metabolite m ='+str(m)+' of '+str(len(met_names)))
        met = met_names[m] #sigBH0[3]

        pval = np.zeros(imax+1)
        power = np.zeros(imax+1)
        for i in range(imax+1):
            f = flist[i]

            nr = Rsubsamples
            ln_nent = Rsubsamples*entropy(f)
            
            if np.log(Rsubsamples)>ln_nent:
                nr = np.floor(ln_nent).astype(int)
            nr = np.max([nr,1])
            
            lnpi = np.zeros(nr)
            power_r = 0
            for j in range(nr):
                n0temp = np.int(np.round(f*n0))
                n1temp = np.int(np.round(f*n1))
                n2temp = np.int(np.round(f*n2))

                a = features[[met]][features['class'] == 0].sample(n=n0temp)[met]
                b = features[[met]][features['class'] == 1].sample(n=n1temp)[met]
                c = features[[met]][features['class'] == 2].sample(n=n2temp)[met]

                #print([i,a,b,c])
                #print(c)
                if SigMethod == "ANOVA":
                    s,p = f_oneway(a,b,c)
                if SigMethod == "KW":
                    s,p = kruskal(a,b,c)
                
                if p <= alpha:
                    power_r = power_r + 1
                    
                lnpi[j] = np.log(p) #p*len(met_names)   

            pval[i] = np.mean(lnpi)
            power[i] = power_r/nr
            
        # - Fitting a linear dependence 
        X = smapi.add_constant(flist)
        y = pval

        model = smapi.OLS(y,X)
        results = model.fit()

        slope_Lnpval[m] = results.params[1]

        ciL_slope_Lnpval[m] = results.conf_int()[1][0]
        ciU_slope_Lnpval[m] = results.conf_int()[1][1]

        intercept_Lnpval[m] = results.params[0]

        ciL_intercept_Lnpval[m] = results.conf_int()[0][0]
        ciU_intercept_Lnpval[m] = results.conf_int()[0][1]

        R2[m] = results.rsquared
        #pvalb[m] = results.pvalues[1]
        
        # One-sided p-value for the slope (negative slope is significant)
        df = results.df_resid
        pvalb[m] = ss.t.cdf(results.tvalues[1], df)
        
        #----
        #------ Linear fit to ln(beta) vs f. beta = 1-power is the false negative rate
        #----
        beta = 1-power
        
        #print(beta)
        
        X = smapi.add_constant(np.array(flist)[np.array(flist)<th_beta])
        y = np.log(beta[np.array(flist)<th_beta])
    
        model = smapi.OLS(y,X,missing='drop')
        results = model.fit()
        #
        df = results.df_resid
        pvalb_beta[m] = ss.t.cdf(results.tvalues[1], df)
        
    return R2,slope_Lnpval,ciL_slope_Lnpval,ciU_slope_Lnpval,intercept_Lnpval,ciL_intercept_Lnpval,ciU_intercept_Lnpval,pvalb,pvalb_beta
