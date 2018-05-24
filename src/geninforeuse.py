import numpy as np
import copy
import pdb
import random as rdm
import time
import scipy.special as scp
import scipy.stats as scs
import scipy.optimize as scopt
import matplotlib.pyplot as plt
import os
import datetime
from scipy import optimize as scipyopt

import utilities as utils

from loggedopt import Log
from alginforeuse import InformationReuseBase

class GeneralInformationReuse(InformationReuseBase):

    def __init__(self, fobj, estimator, **kwargs):
        self.estimator = estimator
        self.n_boot = kwargs.setdefault('n_boot', 1000)
        InformationReuseBase.__init__(self, fobj, **kwargs)
        self.checkAdditionalInputs()


    def checkAdditionalInputs(self):
        try:
            vqsamples = [float(np.random.random(1)) for i in range(self.n_init)]
            dhat = float(self.estimator(vqsamples))
        except TypeError:
            raise Exception('''Estimator must be of the form f(vq) where vq
            is a vector of samples of q and must return a single output
            that is the value of the estimator with these samples.''')


    def reSample(self, sample):

        M = len(sample)
        resample = []
        vrandj = [int(j) for j in np.floor(np.random.random(M)*M)]
        for randj in vrandj:
            resample.append(sample[randj])

        return resample


    def naiveMC(self, vqsamp):

        M = len(vqsamp)
        dhat = self.estimator(vqsamp)

        dhat_resampled = []
        for ii in range(self.n_boot):
            vq_resampled = self.reSample(vqsamp)
            dhat_resampled.append(self.estimator(vq_resampled))

        dvar = np.var(dhat_resampled)

        return {'dhat': dhat, 'dvar': dvar,
                'gamma': 0, 'M':len(vqsamp), 'mx_var': dvar,
                'rho': 0}


    def reuseMC(self, vqx, vqc, cdict):

        M = len(vqx)
        samples = len(vqx) + len(vqc)
        mx = self.estimator(vqx)
        mc = self.estimator(vqc)

        sc = cdict['dhat']
        sc_var = cdict['dvar']

        xd_resampled, cd_resampled = np.zeros(self.n_boot), np.zeros(self.n_boot)
        for ii in range(self.n_boot):
            vqx_resampled, vqc_resampled = np.zeros(M), np.zeros(M)

            i_resampled = self.reSample(range(M))
            for jj, ind in enumerate(i_resampled):
                vqx_resampled[jj], vqc_resampled[jj] = vqx[ind], vqc[ind]

            xd_resampled[ii] = self.estimator(vqx_resampled)
            cd_resampled[ii] = self.estimator(vqc_resampled)

        covar = np.cov(np.array([xd_resampled, cd_resampled]))
        mx_var, mc_var, mxc_cov = covar[0,0], covar[1,1], covar[0,1]

        ## Evaluate gamma
        gam = mxc_cov / (sc_var + mc_var)

        sx = mx + gam*(sc - mc)
        sx_var = mx_var + gam**2 * (sc_var + mc_var) - 2*gam*mxc_cov

        rho2 = covar[0,1]**2/(covar[0,0]*covar[1,1])

        return {'dhat': sx, 'dvar': sx_var, 'M': M,
                'gamma': gam, 'rho': np.sqrt(rho2),
                'mx_var': mx_var, 'mxc_cov': mxc_cov, 'mc_var': mc_var,
                'xd_resampled': xd_resampled, 'cd_resampled': cd_resampled}

    def predictorIR(self, xdict, cdict):

        mcx_cov = xdict['mxc_cov']
        vmc = xdict['mc_var']
        vmx = xdict['mx_var']

        vsc = cdict['dvar']

        n0 = float(xdict['M'])
        def predvar(n):
            n = float(n)
            return vmx*n0/n - (mcx_cov*n0/n)**2 / (vsc+vmc*n0/n) - self.var_req

        ans, info, ier, mesg  = scopt.fsolve(predvar, n0, full_output=True)
        if ier ==1 :
            if self.verbose:
                print 'Predictor converged'
        else:
            if self.verbose:
                print 'Predictor solver did not converge: ', mesg
            ans = int(n0)

        return int(ans)
