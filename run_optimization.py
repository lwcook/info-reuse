#!/usr/bin/python
import numpy as np
import scipy.optimize as scopt

import src.geninforeuse as gir
import src.setup as setup

def deterministic_objective(x, u):
######################################################################
# Change this to be the objective function for a new model
# It takes two inputs: the design variables as the vector x, and
# uncertain input parameters as the vector u
######################################################################

    qrosen = ((2-x[0])**2 + 2*(x[1] - x[0]**2)**2)

    qu = 0.0
    for ii, ui in enumerate(u):
        s = x[0] if ii % 2 == 0 else x[1]
        qu += (1./(ii+1.)) * (1 + 0.2*(s)**2) * ui

    return qrosen - np.exp(qu)



def deterministic_constraint(x, u):
######################################################################
# Change this to be a constraint function for a new model
# It takes two inputs: the design variables as the vector x, and
# uncertain input parameters as the vector u
######################################################################

    qu = 0.0
    for ii, ui in enumerate(u):
        s = x[0] if ii % 2 == 0 else x[1]
        qu += (1./(ii+1.)) * (0.2 + 0.2*(s+1)**2) * ui

    return 2.0 - np.exp(qu)

######################################################################
# Define any more constraints as required
######################################################################


def runopt(IRobj, IRconstr, x0, lb, ub):

    def fopt(x):
        print '-'*70
        print 'Evaluating objective'
        hmdict = IRobj.iteration(x)
        return hmdict['dhat']

    def gopt(x):
        print '-'*70
        print 'Evaluating constraint'
        gdict = IRconstr.iteration(x)
        g = gdict['dhat']
        return g

    constraints = [{'type': 'ineq', 'fun': gopt}]
    for ii, (lbi, ubi) in enumerate(zip(lb, ub)):
        constraints.append({'type': 'ineq', 'fun': lambda x: x[ii] - lbi})
        constraints.append({'type': 'ineq', 'fun': lambda x: -1*x[ii] + ubi})

    scopt.minimize(fopt, x0, method='COBYLA',
            constraints=constraints,
            options={'maxiter': 25})


def main(case='Test'):

    nu = 10  # Dimension of uncertainty space
    nx = 2   # Dimension of design space
    lb = [-2, -2]  # Lower bounds on design space
    ub = [2, 2]  # Lower bounds on design space

    x0 = [lb[i] + (ub[i]-lb[i]) for i in range(nx)] # Initial design point

    n_boot = 2000  # Number of bootstrap resampels to perform
    n_init = 50  # Initial number of samples used at each iteration

    ######################################################################
    # Change these required variances for the problem at hand.
    # Var_obj is required variance by an estimator of the objective
    # Var_constraint is required variance by an estimator of the constraint
    ######################################################################
    var_obj = 50e-4
    var_constr = 40e-4

    ######################################################################
    # Define the statistical estimators of interest here
    ######################################################################
    def obj_estimator(samples):
        targ = lambda h: setup.custom_target(h, q0=0, q1=-4, p=6)
        return setup.hm_estimator(samples, targ=targ),

    def constr_estimator(samples)
        return setup.quantile_estimator(samples, 0.9),

    ## Using information reuse
    theIRobj = gir.GeneralInformationReuse(fobj=deterministic_objective,
              udim=nu, xdim=nx,
              estimator=obj_estimator,
              n_init=n_init, n_boot=n_boot, var_req=var_obj, check=False,
              verbose=True, log_name=case+'_log_IR_objective')

    theIRconstr = gir.GeneralInformationReuse(fobj=deterministic_constraint,
              udim=nu, xdim=nx,
              estimator=constr_estimator
              n_init=n_init, n_boot=n_boot, var_req=var_constr, check=False,
              verbose=True, log_name=case+'_log_IR_constraint')

    ######################################################################
    # Define additional information reuse objects for additional constraints
    ######################################################################

    runopt(theIRobj, theIRconstr, x0, lb, ub)

    ## Only using naive MC
    theIRobj.reset(log_name=case+'_log_MC_objective', verbose=True, bMC=True)
    theIRconstr.reset(log_name=case+'_log_MC_constraint', verbose=True, bMC=True)
    runopt(theIRobj, theIRconstr, x0, lb, ub)


if __name__ == "__main__":
    main()
