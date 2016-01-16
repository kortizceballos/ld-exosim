import numpy as np
import lmfit
import sys
import batman
import LDC3

def get_transit_duration(P, p, r_a, inclination):
    """
    Return the transit duration given the period (P), planet-to-star radius ratio (p = Rp/R_*), 
    the stellar-to-semi-major axis ratio (r_a = R_*/a) and the inclination in radians.
    """


    b = np.cos(inclination)/r_a
    num1 = ( 1. + p )**2 - b**2
    den1 = 1. - np.cos(inclination)**2

    return (P/np.pi)*np.arcsin(r_a * np.sqrt(num1/den1))

def init_batman(times, Period, inclination, a, p, t0, ld_coeffs, e=0.0, omega=0.0, ld_law= 'logarithmic',max_err=0.01):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = Period
    params.rp = p
    params.a = a
    params.inc = inclination
    params.ecc = e
    params.w = omega
    if ld_law == 'linear':
       params.u = [ld_coeffs[0]]
       params.limb_dark = ld_law
       m = batman.TransitModel(params, times)
    elif ld_law == 'quadratic':
       params.u = [ld_coeffs[0],ld_coeffs[1]]
       params.limb_dark = ld_law
       m = batman.TransitModel(params, times)
    elif ld_law == 'logarithmic' or ld_law == 'exponential' or ld_law == 'squareroot':
       params.u = [ld_coeffs[0],ld_coeffs[1]]
       params.limb_dark = ld_law
       m = batman.TransitModel(params, times, max_err = max_err)
    elif ld_law == 'three-param':
       params.u = [0.,ld_coeffs[0],ld_coeffs[1],ld_coeffs[2]]
       params.limb_dark = 'nonlinear'
       m = batman.TransitModel(params, times, max_err = max_err)
    elif ld_law == 'non-linear':
       params.u = [ld_coeffs[0],ld_coeffs[1],ld_coeffs[2],ld_coeffs[3]]
       params.limb_dark = 'nonlinear'
       m = batman.TransitModel(params, times, max_err = max_err)
    return params,m

################### FITTING OF TLs     ###################################

def fit_transit_floating_lds(x, y, guess_p, guess_coeff1, guess_coeff2, guess_i, guess_a, P, t0, ld_law, guess_coeff3 = None):
    """
    This function fits transit lightcurves with floating limb-darkening coefficients. Inputs are:

    x:              Times of observation.

    y:              Normalized fluxes

    guess_p:        Guess for the planet-to-star radius ratio (R_p/R_*)

    guess_coeff1:   Guess for the first limb-darkening coefficient.

    guess_coeff2:   Same for the second coefficient (not used if law is linear).

    guess_coeff3:   Same for third coefficient (if non-linear law is used)

    guess_i:        Guess on the inclination.

    guess_a:        Guess on the scaled semi-major axis (a/R_*)

    P:              Period of the orbit (not fitted).

    t0:             Time of transit center of the orbit (not fitted).

    ld_law:         Limb-darkening law to use for the fit.

    The function returns the fitted values of p,i,a and the coefficients of the chosen LD law.
    """

    # First, initialize the batman transit modeller with guess parameters:
    if ld_law == 'linear':
                batman_params,batman_m = init_batman(x, P, guess_i, guess_a, guess_p, t0, \
                                             [guess_coeff1], ld_law = ld_law)
    elif ld_law == 'three-param':
                batman_params,batman_m = init_batman(x, P, guess_i, guess_a, guess_p, t0, \
                                             [guess_coeff1, guess_coeff2,guess_coeff3], ld_law = ld_law)
    else:
                batman_params,batman_m = init_batman(x, P, guess_i, guess_a, guess_p, t0, \
                                             [guess_coeff1, guess_coeff2], ld_law = ld_law)

    # Define the residual functions depending on the LD law used:
    def residuals_linear(params, x, y):

        batman_params.rp = params['p'].value
        batman_params.a = params['a'].value
        batman_params.inc = params['i'].value
        batman_params.u = [params['q1'].value]
        return y - batman_m.light_curve(batman_params)
    
    def residuals_quadratic(params, x, y):

        coeff1 = 2.*np.sqrt(params['q1'].value)*params['q2'].value
        coeff2 = np.sqrt(params['q1'].value)*(1.-2.*params['q2'].value)

        batman_params.rp = params['p'].value
        batman_params.a = params['a'].value
        batman_params.inc = params['i'].value
        batman_params.u = [coeff1,coeff2]
        return y - batman_m.light_curve(batman_params)

    def residuals_sqroot(params, x, y):

        coeff1 = np.sqrt(params['q1'].value)*(1.-2.*params['q2'].value)
        coeff2 = 2.*np.sqrt(params['q1'].value)*params['q2'].value

        batman_params.rp = params['p'].value
        batman_params.a = params['a'].value
        batman_params.inc = params['i'].value
        batman_params.u = [coeff1,coeff2]

        return y - batman_m.light_curve(batman_params)

    def residuals_logarithmic(params, x, y):

        coeff1 = 1.-np.sqrt(params['q1'].value)*params['q2'].value
        coeff2 = 1.-np.sqrt(params['q1'].value)

        batman_params.rp = params['p'].value
        batman_params.a = params['a'].value
        batman_params.inc = params['i'].value
        batman_params.u = [coeff1,coeff2]

        return y - batman_m.light_curve(batman_params) 

    def residuals_three_param(params, x, y):

        c = LDC3.forward([params['q1'].value,params['q2'].value, params['q3'].value])

        if LDC3.criteriatest(0,c) == 0:
           c = [guess_coeff1, guess_coeff2, guess_coeff3]

        batman_params.rp = params['p'].value
        batman_params.a = params['a'].value
        batman_params.inc = params['i'].value
        batman_params.u = [0.,c[0],c[1],c[2]]

        return y - batman_m.light_curve(batman_params)

    # Define the initial values of the q_i parameters with 
    # the initial guesses of the coefficients:
    if ld_law == 'linear':
        guess_q1 = guess_coeff1

    elif ld_law == 'quadratic':
        guess_q1 = (guess_coeff1 + guess_coeff2)**2
        guess_q2 = (guess_coeff1)/(2.*(guess_coeff1+guess_coeff2))

    elif ld_law == 'squareroot':
        guess_q1 = (guess_coeff1 + guess_coeff2)**2
        guess_q2 = (guess_coeff2)/(2.*(guess_coeff1+guess_coeff2))

    elif ld_law == 'logarithmic':
        guess_q1 = (1.-guess_coeff2)**2
        guess_q2 = (1.-guess_coeff1)/(1.-guess_coeff2)

    elif ld_law == 'three-param':
        guess_q1,guess_q2,guess_q3 = LDC3.inverse([guess_coeff1,guess_coeff2,guess_coeff3])

    # Init lmfit
    prms = lmfit.Parameters()
    prms.add('p', value = guess_p, min = 0, vary = True)
    prms.add('q1', value = guess_q1, min = 0, max = 1, vary = True)
    if ld_law != 'linear':
       prms.add('q2', value = guess_q2, min = 0, max = 1, vary = True)
    if ld_law == 'three-param':
       prms.add('q3', value = guess_q2, min = 0, max = 1, vary = True)
    prms.add('i', value = guess_i, min = 0, max = 90, vary = True)
    if guess_i == 90.0:
       prms.add('a', value = guess_a, min = 0, vary = True)
    else:
       prms.add('b', value = np.cos(guess_i*np.pi/180.0)*guess_a, min = 0, max = 1, vary = True)
       prms.add('a', expr = 'b/cos(i * pi/180.0)')

    # Run lmfit for the corresponding ld-law:
    if ld_law == 'linear':
        result = lmfit.minimize(residuals_linear, prms, args=(x,y))
        coeff1out = result.params['q1'].value

    elif ld_law == 'quadratic':
        result = lmfit.minimize(residuals_quadratic, prms, args=(x,y))
        coeff1out = 2.*np.sqrt(result.params['q1'].value)*result.params['q2'].value
        coeff2out = np.sqrt(result.params['q1'].value)*(1.-2.*result.params['q2'].value)

    elif ld_law == 'squareroot':
        result = lmfit.minimize(residuals_sqroot, prms, args=(x,y))
        coeff1out = np.sqrt(result.params['q1'].value)*(1.-2.*result.params['q2'].value)
        coeff2out = 2.*np.sqrt(result.params['q1'].value)*result.params['q2'].value

    elif ld_law == 'logarithmic':
        result = lmfit.minimize(residuals_logarithmic, prms, args=(x,y))
        coeff1out = 1.-np.sqrt(result.params['q1'].value)*result.params['q2'].value
        coeff2out = 1.-np.sqrt(result.params['q1'].value)

    elif ld_law == 'three-param':
        result = lmfit.minimize(residuals_three_param, prms, args=(x,y))
        coeff1out, coeff2out, coeff3out = LDC3.forward([result.params['q1'].value,\
                                          result.params['q2'].value,result.params['q3'].value])

    # If the fit is successful, return fitted parameters. If not, raise an
    # message and end the program:
    if result.success:
       if ld_law == 'linear':
          return result.params['p'].value, coeff1out,  result.params['i'].value, result.params['a'].value
       elif ld_law == 'three-param':
          return result.params['p'].value, coeff1out, coeff2out, coeff3out, result.params['i'].value, result.params['a'].value
       else:
          return result.params['p'].value, coeff1out, coeff2out, result.params['i'].value, result.params['a'].value
    else:
       print 'Unsuccessful fit for LD law',ld_law
       if ld_law == 'linear':
          return guess_p, guess_coeff1,  guess_i, guess_a
       elif ld_law == 'three-param':
          return guess_p, guess_coeff1, guess_coeff2, guess_coeff3, guess_i, guess_a
       else:
          return guess_p, guess_coeff1, guess_coeff2, guess_i, guess_a

def fit_transit_fixed_lds(x, y, guess_p, coeff1, coeff2, guess_i, guess_a, P, t0, ld_law, coeff3 = None):
    """
    This function fits transit lightcurves with fixed limb-darkening coefficients. Inputs are:

        x:              Times of observation.

        y:              Normalized fluxes

        guess_p:        Guess for the planet-to-star radius ratio (R_p/R_*)

        coeff1:   	    First fixed limb-darkening coefficient.

        coeff2:  	    Same for the second coefficient (not used if law is linear).

        coeff3:         Same for third coefficient (if non-linear law is used)

        guess_i:        Guess on the inclination.

        guess_a:        Guess on the scaled semi-major axis (a/R_*)

        P:              Period of the orbit (not fitted).

        t0:             Time of transit center of the orbit (not fitted).

        ld_law:         Limb-darkening law to use for the fit.

    The function returns the fitted values of p,i,a.
    """

    # First, initialize the batman transit modeller with guess parameters:
    if ld_law == 'linear':
                batman_params,batman_m = init_batman(x, P, guess_i, guess_a, guess_p, t0, \
                                             [coeff1], ld_law = ld_law)
    elif ld_law == 'three-param':
                batman_params,batman_m = init_batman(x, P, guess_i, guess_a, guess_p, t0, \
                                             [coeff1, coeff2, coeff3], ld_law = ld_law)
    else:
                batman_params,batman_m = init_batman(x, P, guess_i, guess_a, guess_p, t0, \
                                             [coeff1, coeff2], ld_law = ld_law)
    # Define the residual functions. Because the initialization contains the information 
    # about the used LD law, there is no need to create a different one for each LD law: 
    def residuals(params, x, y):

        batman_params.rp = params['p'].value
        batman_params.a = params['a'].value
        batman_params.inc = params['i'].value

        return y - batman_m.light_curve(batman_params)

    # Init lmfit
    prms = lmfit.Parameters()
    prms.add('p', value = guess_p, min = 0, vary = True)
    prms.add('i', value = guess_i, min = 0, max = 90, vary = True)
    if guess_i == 90.0:
       prms.add('a', value = guess_a, min = 0, vary = True)
    else:
       prms.add('b', value = np.cos(guess_i*np.pi/180.0)*guess_a, min = 0, max = 1, vary = True)
       prms.add('a', expr = 'b/cos(i * pi/180.0)')

    # Run lmfit:
    result = lmfit.minimize(residuals, prms, args=(x,y))

    # If the fit is successful, return fitted parameters. If not, raise an
    # message and end the program:
    if result.success:
       return result.params['p'].value, result.params['i'].value, result.params['a'].value
    else:
       print 'Unsuccessful fit for LD law',ld_law
       return guess_p, guess_i, guess_a

################### READING OF LD DATA ###################################
def closest_stellar_params(teff,logg,feh,vturb,input_teff,input_logg,input_feh,input_vturb):
    if input_teff is None:
        idx_teff = (np.where(teff == 5500.)[0])[0]
    else:
        min_dist_teff = np.min(np.abs(teff-input_teff))
        idx_teff = np.where(np.abs(teff-input_teff) == min_dist_teff)[0]
        if len(idx_teff)>1:
           idx_teff = idx_teff[0]
    min_dist_logg = np.min(np.abs(logg-input_logg))
    idx_logg = np.where(np.abs(logg-input_logg) == min_dist_logg)[0]
    if len(idx_logg)>1:
       idx_logg = idx_logg[0]  
    min_dist_feh = np.min(np.abs(feh-input_feh))
    idx_feh = np.where(np.abs(feh-input_feh) == min_dist_feh)[0]
    if len(idx_feh)>1:
       idx_feh = idx_feh[0]
    min_dist_vturb = np.min(np.abs(vturb-input_vturb))
    idx_vturb = np.where(np.abs(vturb-input_vturb) == min_dist_vturb)[0]
    if len(idx_vturb)>1:
       idx_vturb = idx_vturb[0]
    return teff[idx_teff],logg[idx_logg],feh[idx_feh],vturb[idx_vturb]

def read_ld_table(law, input_teff = None, input_logg = 4.5, input_feh = 0.0, input_vturb = 2.0, max_teff = 9000., table_name = 'kepler_atlas_lds.dat'):
    if law == 'linear':
        teff,logg,feh,vturb,a = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,4,5,6,7))
        in_teff,input_logg,input_feh,input_vturb = closest_stellar_params(teff,logg,feh,vturb,input_teff,input_logg,input_feh,input_vturb)
        if input_teff is None:
            idx = np.where((teff < max_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return teff[idx],a[idx]
        else:
            idx = np.where((teff == in_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return [a[idx]]
    if law == 'quadratic':
        teff,logg,feh,vturb,u1,u2 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,4,5,6,8,9))
        in_teff,input_logg,input_feh,input_vturb = closest_stellar_params(teff,logg,feh,vturb,input_teff,input_logg,input_feh,input_vturb)
        if input_teff is None:
            idx = np.where((teff < max_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return teff[idx],u1[idx],u2[idx]
        else:
            idx = np.where((teff == in_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return [u1[idx],u2[idx]]
    if law == 'three-param':
        teff,logg,feh,vturb,b1,b2,b3 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,4,5,6,10,11,12))
        in_teff,input_logg,input_feh,input_vturb = closest_stellar_params(teff,logg,feh,vturb,input_teff,input_logg,input_feh,input_vturb)
        if input_teff is None:
            idx = np.where((teff < max_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return teff[idx],b1[idx],b2[idx],b3[idx]
        else:
            idx = np.where((teff == in_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return [b1[idx],b2[idx],b3[idx]]
    if law == 'non-linear':
        teff,logg,feh,vturb,c1,c2,c3,c4 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,4,5,6,13,14,15,16))
        in_teff,input_logg,input_feh,input_vturb = closest_stellar_params(teff,logg,feh,vturb,input_teff,input_logg,input_feh,input_vturb)
        if input_teff is None:
            idx = np.where((teff < max_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return teff[idx],c1[idx],c2[idx],c3[idx],c4[idx]
        else:
            idx = np.where((teff == in_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return [c1[idx],c2[idx],c3[idx],c4[idx]]
    if law == 'logarithmic':
        teff,logg,feh,vturb,l1,l2 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,4,5,6,17,18))
        in_teff,input_logg,input_feh,input_vturb = closest_stellar_params(teff,logg,feh,vturb,input_teff,input_logg,input_feh,input_vturb)
        if input_teff is None:
            idx = np.where((teff < max_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return teff[idx],l1[idx],l2[idx]
        else:
            idx = np.where((teff == in_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return [l1[idx],l2[idx]]
    if law == 'exponential':
        teff,logg,feh,vturb,e1,e2 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,4,5,6,19,20))
        in_teff,input_logg,input_feh,input_vturb = closest_stellar_params(teff,logg,feh,vturb,input_teff,input_logg,input_feh,input_vturb)
        if input_teff is None:
            idx = np.where((teff < max_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return teff[idx],e1[idx],e2[idx]
        else:
            idx = np.where((teff == in_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return [e1[idx],e2[idx]]
    if law == 'squareroot':
        teff,logg,feh,vturb,s1,s2 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,4,5,6,21,22))
        in_teff,input_logg,input_feh,input_vturb = closest_stellar_params(teff,logg,feh,vturb,input_teff,input_logg,input_feh,input_vturb)
        if input_teff is None:
            idx = np.where((teff < max_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]  
            return teff[idx],s1[idx],s2[idx]
        else:
            idx = np.where((teff == in_teff)&(logg == input_logg)&(feh == input_feh)&(vturb == input_vturb))[0]
            return [s1[idx],s2[idx]]

    print 'Limb-darkening law '+law+' not supported.' 
    sys.exit()

