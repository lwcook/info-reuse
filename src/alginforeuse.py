import numpy as np
import copy
import pdb
import random as rdm
import time
import scipy.special as scp
import scipy.stats as scs
import scipy.optimize as scopt
import matplotlib.pyplot as plt
from scipy import optimize as scipyopt
import datetime

import utilities as utils
from loggedopt import Log


class InformationReuseBase(Log):
    '''Base class for information reuse that deals with setup of the problem
    and checking the inputs, and common functions. 
    Does not include any optimization logic

    Note that algorithm specific functions return dictionaries, so that as long
    as the dictionary has the requried keys it will still work. This way extra
    stuff can be added to be logged without having to change function
    definitions etc. 
    '''

    def __init__(self, fobj, **kwargs):
        '''
        - fobj: quantity of interest.
        - udim: number of uncertainties
        - n_mc, n_quad: no. sample and quadrature points in HM'''
        # Note: notation
        # prefix of v indicates a vector
        # prefix of m indicates a matrix
        # prefix of f indicates a function

        log_name = kwargs.setdefault('log_name', None)
        Log.__init__(self, log_name=log_name)

        self.fobj = fobj  # fobj should take two inputs: dv, u.
        self.udim = kwargs.setdefault('udim', 1)
        self.xdim = kwargs.setdefault('xdim', 1)

        self.n_init = int(kwargs.setdefault('n_init', 50))
        self.n_boot = int(kwargs.setdefault('n_boot', 1e3))
        self.var_req = float(kwargs.setdefault('var_req', 1e-4))
        self.verbose = kwargs.setdefault('verbose', False)

        self.bMC = kwargs.setdefault('bMC', False)
        self.bPlot = kwargs.setdefault('bPlot', False)

        self.bValidation = kwargs.setdefault('bValidation', False)
        self.n_fixed = int(kwargs.setdefault('n_fixed', 1000))

        self.iters = 0
        self.db = []
        self.total_samples = 0
        self.checkInputs()
        if kwargs.setdefault('check_overwrite', False):
            self.overwriteLogFile(self.log_file)
        else:
            with open(self.log_file, 'w') as f: pass


    def reset(self, **kwargs):

        for k, v in kwargs.iteritems():
            if k in dir(self):
                setattr(self, k, v)
            else:
                raise KeyError('reset argument not an object attribute')

        Log.__init__(self, log_name=self.log_name)

        self.iters = 0
        self.total_samples = 0
        self.db = []
        self.checkInputs()
        if kwargs.setdefault('check_overwtire', False):
            self.overwriteLogFile(self.log_file)
        else:
            with open(self.log_file, 'w') as f: pass


    def checkInputs(self):
        try:
            if self.verbose:
                print 'Checking input function with zero vectors'
            q = self.fobj(np.zeros(self.xdim), np.zeros(self.udim))
        except TypeError:
            raise TypeError('''Something is wrong with the objective function,
                Objective function must be of the form f(x, u)
                and return a single output q, the quantity of interest''')


    def evalq(self, fx, vx, vu):

        vqsamp = np.zeros(vu.size / self.udim)
        for ii, ui in enumerate(vu):
            vqsamp[ii] = fx(vx, ui)

        return vqsamp


    def iteration(self, vx):

        if self.verbose:
            print '____________________________________________________________'
            print 'Design: ', vx
            print '____________________________________________________________'

        if not self.db:
            if self.bValidation:
                ddict = self.runValidationIteration(vx, {}, reuse=False)
            else:
                ddict = self.runFirstIteration(vx)
        else:
            vd = np.zeros(len(self.db))
            for il in range(len(self.db)):
                ldict = copy.copy(self.db[il])
                vl = np.array(ldict['design'])
                vd[il] = np.linalg.norm(vl - np.array(vx))
            ilmin = np.argmin(vd)
            cdict = copy.copy(self.db[ilmin])

            if self.bValidation:
                ddict = self.runValidationIteration(vx, cdict, reuse=True)
            else:
                ddict = self.runGeneralIteration(vx, cdict)

        self.total_samples += ddict['samples']
        self.db.append(ddict)
        if ddict is not None:
            self.writeToLog(ddict)
        return ddict


    def runValidationIteration(self, vx, cdict, **kwargs):

        M = self.n_fixed
        vunew = self.sampleUncertainties(M)
        vqx = self.evalq(self.fobj, vx, vunew)
        vqc = []

        if not kwargs.setdefault('reuse', False):
            outdict = self.naiveMC(vqx)
        else:
            if self.bMC:
                outdict = self.naiveMC(vqx)
            else:
                vqc = self.evalq(self.fobj, cdict['design'], vunew)
                outdict = self.reuseMC(vqx, vqc, cdict)

        iterdict = { k: v for k, v in outdict.iteritems()}
        iterdict['design'] =  [x for x in vx]
        iterdict['samples'] = len(vqx) + len(vqc)
        return iterdict


    def runFirstIteration(self, vx, **kwargs):

        iteration = 0
        vqx = []
        M, predMC = self.n_init/1.1, self.n_init/1.1

        for guesses in range(20):

            Mnew = max(int(1.1*predMC), int(1.1*len(vqx)))
            if len(vqx) > 0:
                Mnew = min(Mnew, 10*len(vqx))

            if self.verbose:
                print 'Total ', Mnew, ' samples'

            vunew = self.sampleUncertainties(int(Mnew)-len(vqx))
            vqx = np.concatenate([vqx, self.evalq(self.fobj, vx, vunew)])
            outdict = self.naiveMC(vqx)

            predMC = len(vqx)*outdict['mx_var']/self.var_req

            if self.verbose:
                print 'Variance: ', outdict['dvar']

            if outdict['dvar'] < self.var_req:
                break

        if self.verbose:
            print 'Estimator value: ', outdict['dhat'], '   variance', outdict['dvar']

        iterdict = { k: v for k, v in outdict.iteritems()}
        iterdict['samples'] = len(vqx)
        iterdict['rho_samples'] = 0
        iterdict['design'] =  [x for x in vx]
        iterdict['iteration'] = copy.copy(iteration)
        return iterdict

    def sampleUncertainties(self, num):
        return  -1. + 2.*np.random.random((num, self.udim))
#       return 1 - np.random.gamma(2., 0.5, size=(num, self.udim))
#        return np.random.randn(num, self.udim)

        x1 = np.random.beta(2, 3, (num, self.udim))
        x2 = -1 + np.random.beta(2, 3, (num, self.udim))
        b1 = np.random.binomial(1, 0.5, num)
        s = np.zeros((num, self.udim))
        for ib, b in enumerate(b1):
            if b == 1:
                s[ib, :] = x1[ib, :]
            else:
                s[ib, :] = x2[ib, :]

        return np.array(s)


    def runGeneralIteration(self, vx, cdict, **kwargs):

        iteration = self.db[-1]['iteration'] + 1

        vc = cdict['design']
        vqx, vqc, convIR, convMC = [], [], [], []
        predM = self.n_init/1.1
        bMC = copy.copy(self.bMC)
        rho_samples = 0

        for guesses in range(20):

            Mnew = max(int(1.1*predM), int(1.1*len(vqx)))
            if len(vqx) > 0:
                Mnew = min(Mnew, 10*len(vqx))

            if self.verbose:
                print 'Total ', Mnew, ' samples'

            if bMC:

                vunew = self.sampleUncertainties(Mnew - len(vqx))
                vqx = np.concatenate([vqx, self.evalq(self.fobj, vx, vunew)])

                outdict = self.naiveMC(vqx)

                predM = len(vqx)*outdict['mx_var']/self.var_req

            else:

                vunew = self.sampleUncertainties(Mnew - len(vqx))
                vqx = np.concatenate([vqx, self.evalq(self.fobj, vx, vunew)])
                vqc = np.concatenate([vqc, self.evalq(self.fobj, vc, vunew)])
                rho_samples = np.corrcoef(np.array([vqx, vqc]))[0,1]

                outdict = self.reuseMC(vqx, vqc, cdict)

                predNaive = len(vqx)*outdict['mx_var']/self.var_req
                predM = self.predictorIR(outdict, cdict)

                if 2*predM > predNaive:
                    bMC = True
                    predM = int(predNaive)
                    if self.verbose:
                        print 'SWITCHING TO REGULAR MC'

            if self.verbose:
                print 'Estimator value: ', outdict['dhat'], '   variance', outdict['dvar']

            if outdict['dvar'] < self.var_req:
                break

        if self.bPlot:
            self.plotConvergence(convMC, convIR, cdict, vx, self.var_req)

        iterdict = { k: v for k, v in outdict.iteritems()}
        iterdict['samples'] = len(vqx) + len(vqc)
        iterdict['rho_samples'] = rho_samples
        iterdict['design'] = [x for x in vx]
        iterdict['iteration'] = copy.copy(iteration)
        return iterdict


    def plotConvergence(**kwargs):
        raise Exception('Plotting not implemented')


    def naiveMC(**kwargs):
        raise Exception('naiveMC function needs to be implemented')


    def reuseMC(**kwargs):
        raise Exception('reuseMC function needs to be implemented')


    def predictorIR(**kwargs):
        raise Exception('predictorIR function needs to be implemented')



class AlgebraicInformationReuse(InformationReuseBase):
    '''Class that that performs algebraic information reuse from Ng and Willcox
    (2014). Functionality at the moment just uses the mean as the estimator'''

    def __init__(self, fobj, **kwargs):
        self.estimator = kwargs.setdefault('estimator', 'mean')
        self.mvweight = kwargs.setdefault('mvweight', 1.282)
        InformationReuseBase.__init__(self, fobj, **kwargs)
        self.checkAdditionalInputs()


    def checkAdditionalInputs(self):
        el = self.estimator.lower()
        if el != 'mean' and (el != 'var' and el != 'ws'):
            raise ValueError('''estimator for algebraic information reuse should
            be ''mean'' or ''var'' or ''ws''')


    def runFirstIteration(self, vx, **kwargs):
        return InformationReuseBase.runFirstIteration(self, vx, **kwargs)


    def runGeneralIteration(self, vx, cdict, **kwargs):
        return InformationReuseBase.runGeneralIteration(self, vx, cdict, **kwargs)


    def naiveMC(self, vqsamp):

        mv, mv_cov = [], []
        if self.estimator.lower() == 'mean':
            dhat = np.mean(vqsamp)
            dvar = np.var(vqsamp)/len(vqsamp)

        elif self.estimator.lower() == 'var':

            n, xsum, xsumm1 = len(vqsamp), sum(vqsamp), sum(vqsamp[0:-1])
            vvsamp = [(n/(n-1.))*(q - (1./n)*xsum)*(q - (1./(n-1.))*xsumm1)
                        for q in vqsamp]

            dhat = np.mean(vvsamp)
            dvar = np.var(vvsamp)/len(vvsamp)

        elif self.estimator.lower() == 'ws':
            n, qsum, qsumm1 = len(vqsamp), sum(vqsamp), sum(vqsamp[0:-1])
            vvsamp = [(n/(n-1.))*(q - (1./n)*qsum)*(q - (1./(n-1.))*qsumm1)
                        for q in vqsamp]

            m = np.mean(vqsamp)
            v = np.mean(vvsamp)
            w = self.mvweight
            dhat = m + w*np.sqrt(v)

            mv_cov = (1./n)*np.cov(np.array(zip(vqsamp, vvsamp)).T)

            fgrad = np.array([1, self.mvweight/(2.*np.sqrt(v))]).reshape([2,1])
            dvar = float(fgrad.T.dot(mv_cov.dot(fgrad)))

            mv = np.array((m, v))

        return {'dhat': dhat, 'dvar': dvar,
                'gamma': 0, 'eta': 0, 'M':len(vqsamp),
                'mx_var': dvar, 'mv_cov': mv_cov, 'mv': mv,
                'rho': 0}


    def _getreuseterms(self, vqx, vqc, sc_var):

        M = len(vqx)

        abar = np.mean(vqx)
        cbar = np.mean(vqc)
        asum2 = sum([(ai-abar)**2 for ai in vqx])
        csum2 = sum([(ci-cbar)**2 for ci in vqc])
        acsum = sum([(ai - abar)*(ci - cbar) for ai, ci in zip(vqx, vqc)])

        rho2 = acsum**2 / (asum2 * csum2)
        eta = sc_var*M*(M-1) / csum2
        gam = (1./(1. + eta)) * acsum / csum2

        siga2 = (1./(M-1)) * asum2
        sigc2 = (1./(M-1)) * csum2

        return gam, eta, rho2, siga2, sigc2


    def reuseMC(self, vqx, vqc, cdict):
        '''Uses the method outlined in Ng and Willcox (2014) Journal of
        Aircraft for information re-use'''

        # Notation:
        #  sx - estimator at point x
        #  sc - estimator at point c
        #  mx - sample average at point x
        #  mc - sample average at point c

        M = len(vqx)
        samples = len(vqx) + len(vqc)
        mv, mv_cov, sc_cov = [], [], []

        if self.estimator.lower() == 'mean':

            sc = cdict['dhat']
            sc_var = cdict['dvar']

            abar, cbar = np.mean(vqx), np.mean(vqc)
            gam, eta, rho2, siga2, sigc2 = self._getreuseterms(vqx, vqc, sc_var)

            dhat = abar + gam*(sc - cbar)
            dvar = (1./M)*( siga2 + gam**2*sigc2*(1. + eta) - \
                    2*gam*np.sqrt(rho2*siga2*sigc2) )

            mx_var = float(siga2/M)
            mc_var = float(sigc2/M)
            mxc_cov = float(rho2*np.sqrt(siga2*sigc2)/M)

        if self.estimator.lower() == 'var':

            sc = cdict['dhat']
            sc_var = cdict['dvar']

            n, xsum, xsumm1 = len(vqx), sum(vqx), sum(vqx[0:-1])
            vvx = [(n/(n-1.))*(q - (1./n)*xsum)*(q - (1./(n-1.))*xsumm1)
                        for q in vqx]
            n, csum, csumm1 = len(vqc), sum(vqc), sum(vqc[0:-1])
            vvc = [(n/(n-1.))*(q - (1./n)*csum)*(q - (1./(n-1.))*csumm1)
                        for q in vqc]

            abar, cbar = np.mean(vvx), np.mean(vvc)
            gam, eta, rho2, siga2, sigc2 = self._getreuseterms(vvx, vvc, sc_var)

            dhat = abar + gam*(sc - cbar)
            dvar = (1./M)*( siga2 + gam**2*sigc2*(1. + eta) - \
                    2*gam*np.sqrt(rho2*siga2*sigc2) )

            mx_var = float(siga2)/M
            mc_var = float(sigc2)/M
            mxc_cov = float(rho2*np.sqrt(siga2*sigc2))/M

        elif self.estimator.lower() == 'ws':

            sc_var = cdict['dvar']
            sc_m, sc_v = cdict['mv'][0], cdict['mv'][1]
            sc_cov = np.array(cdict['mv_cov'])

            n, xsum, xsumm1 = len(vqx), sum(vqx), sum(vqx[0:-1])
            vvx = [(n/(n-1.))*(q - (1./n)*xsum)*(q - (1./(n-1.))*xsumm1)
                        for q in vqx]
            n, csum, csumm1 = len(vqc), sum(vqc), sum(vqc[0:-1])
            vvc = [(n/(n-1.))*(q - (1./n)*csum)*(q - (1./(n-1.))*csumm1)
                        for q in vqc]

            abar_m, cbar_m = np.mean(vqx), np.mean(vqc)
            gam_m, eta_m, rho2_m, siga2_m, sigc2_m = self._getreuseterms(
                    vqx, vqc, sc_cov[0,0])

            m = abar_m + gam_m*(sc_m - cbar_m)
            mvar = (1./n)*( siga2_m + gam_m**2*sigc2_m*(1. + eta_m) - \
                    2*gam_m*np.sqrt(rho2_m*siga2_m*sigc2_m) )

            abar_v, cbar_v = np.mean(vvx), np.mean(vvc)
            gam_v, eta_v, rho2_v, siga2_v, sigc2_v = self._getreuseterms(
                    vvx, vvc, sc_cov[1,1])

            v = abar_v + gam_v*(sc_v - cbar_v)
            vvar = (1./n)*( siga2_v + gam_v**2*sigc2_v*(1. + eta_v) - \
                    2*gam_v*np.sqrt(rho2_v*siga2_v*sigc2_v) )

            xcov = np.cov(np.array(zip(vqx, vvx)).T)
            ccov = np.cov(np.array(zip(vqc, vvc)).T)
            xccrosscov = np.cov(np.array(zip(vqx, vvx, vqc, vvc)).T)[0:2,2:4]
            cxcrosscov = np.cov(np.array(zip(vqx, vvx, vqc, vvc)).T)[2:4,0:2]

            gam = np.array([[gam_m, 0.], [0., gam_v]])

            mv_cov = (1./n)*(xcov + (gam.dot(n*sc_cov + ccov)).dot(gam) -
                    (gam.dot(cxcrosscov) + xccrosscov.dot(gam)))

            mv = np.array([m, v])

            w = self.mvweight
            dhat = m + w*np.sqrt(v)

            fgrad = np.array([1, self.mvweight/(2.*np.sqrt(v))]).reshape([2,1])
            dvar = float(fgrad.T.dot(mv_cov.dot(fgrad)))

            mx_var = float(fgrad.T.dot(xcov.dot(fgrad)))/M
            mc_var = float(fgrad.T.dot(ccov.dot(fgrad)))/M
            mxc_cov = float(fgrad.T.dot(xccrosscov.dot(fgrad)))/M
            eta = cdict['dvar']/mc_var

        return {'dhat': dhat, 'dvar': dvar, 'mv': mv, 'mv_cov': mv_cov,
            'gamma': gam, 'eta': eta, 'M': M, 'rho': 0,
            'mx_var': mx_var, 'mc_var': mc_var, 'mxc_cov': mxc_cov}


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
        if ier != 1:
            if self.verbose:
                print 'Predictor solver did not converge: ', mesg
                print 'Adding a few more samples'
            ans = int(n0)

        return int(ans)
