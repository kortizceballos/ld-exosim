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
    if ld_law == 'quadratic':
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

	x:		Times of observation.

	y:		Normalized fluxes

	guess_p:	Guess for the planet-to-star radius ratio (R_p/R_*)

	guess_coeff1:	Guess for the first limb-darkening coefficient.

	guess_coeff2:	Same for the second coefficient.

	guess_coeff3:   Same for third coefficient (if non-linear law is used)

	guess_i:	Guess on the inclination.

	guess_a:	Guess on the scaled semi-major axis (a/R_*)

	P:		Period of the orbit (not fitted).

	t0:		Time of transit center of the orbit (not fitted).

	ld_law:		Limb-darkening law to use for the fit.

    The function returns the fitted values of p,i,a and the coefficients of the chosen LD law.
    """

    # First, initialize the batman transit modeller with guess parameters:
    if ld_law == 'three-param':
                batman_params,batman_m = init_batman(x, P, guess_i, guess_a, guess_p, t0, \
                                             [guess_coeff1, guess_coeff2,guess_coeff3], ld_law = ld_law)
    else:
    		batman_params,batman_m = init_batman(x, P, guess_i, guess_a, guess_p, t0, \
                		             [guess_coeff1, guess_coeff2], ld_law = ld_law)

    # Define the residual functions depending on the LD law used:
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
    if ld_law == 'quadratic':
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
    if ld_law == 'quadratic':
	result = lmfit.minimize(residuals_quadratic, prms, args=(x,y))
        coeff1out = 2.*np.sqrt(prms['q1'].value)*prms['q2'].value
        coeff2out = np.sqrt(prms['q1'].value)*(1.-2.*prms['q2'].value)

    elif ld_law == 'squareroot':
	result = lmfit.minimize(residuals_sqroot, prms, args=(x,y))
        coeff1out = np.sqrt(prms['q1'].value)*(1.-2.*prms['q2'].value)
        coeff2out = 2.*np.sqrt(prms['q1'].value)*prms['q2'].value

    elif ld_law == 'logarithmic':
	result = lmfit.minimize(residuals_logarithmic, prms, args=(x,y))
    	coeff1out = 1.-np.sqrt(prms['q1'].value)*prms['q2'].value
    	coeff2out = 1.-np.sqrt(prms['q1'].value)

    elif ld_law == 'three-param':
        result = lmfit.minimize(residuals_three_param, prms, args=(x,y))
        coeff1out, coeff2out, coeff3out = LDC3.forward([prms['q1'].value,\
                                          prms['q2'].value,prms['q3'].value])

    # If the fit is successful, return fitted parameters. If not, raise an
    # message and end the program:
    if result.success:
       if ld_law == 'three-param':
          return prms['p'].value, coeff1out, coeff2out, coeff3out, prms['i'].value, prms['a'].value
       else:
          return prms['p'].value, coeff1out, coeff2out, prms['i'].value, prms['a'].value
    else:
       print 'Unsuccessful fit'
       print guess_p, guess_coeff1, guess_coeff2, guess_i, guess_a, P, t0, ld_law
       sys.exit()

def fit_transit_fixed_lds(x, y, guess_p, coeff1, coeff2, guess_i, guess_a, P, t0, ld_law, coeff3 = None):
    """
    This function fits transit lightcurves with fixed limb-darkening coefficients. Inputs are:

        x:              Times of observation.

        y:              Normalized fluxes

        guess_p:        Guess for the planet-to-star radius ratio (R_p/R_*)

        coeff1:   	First fixed limb-darkening coefficient.

        coeff2:  	Same for the second coefficient.

        coeff3:         Same for third coefficient (if non-linear law is used)

        guess_i:        Guess on the inclination.

        guess_a:        Guess on the scaled semi-major axis (a/R_*)

        P:              Period of the orbit (not fitted).

        t0:             Time of transit center of the orbit (not fitted).

        ld_law:         Limb-darkening law to use for the fit.

    The function returns the fitted values of p,i,a.
    """

    # First, initialize the batman transit modeller with guess parameters:
    if ld_law == 'three-param':
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
       return prms['p'].value, prms['i'].value, prms['a'].value
    else:
       print 'Unsuccessful fit'
       print guess_p, guess_i, guess_a, P, t0, ld_law
       sys.exit()

################### READING OF LD DATA ###################################

def read_ld_table(law, table_name = 'espinoza_table.dat'):	
	if law == 'linear':
		teff,a = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,7))
		return teff,a
	if law == 'quadratic':
		teff,u1,u2 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,8,9))
		return teff,u1,u2
	if law == 'three-param':
		teff,b1,b2,b3 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,10,11,12))
		return teff,b1,b2,b3
	if law == 'non-linear':
		teff,c1,c2,c3,c4 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,13,14,15,16))
		return teff,c1,c2,c3,c4
	if law == 'logarithmic':
		teff,l1,l2 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,17,18))
		return teff,l1,l2
	if law == 'exponential':
		teff,e1,e2 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,19,20))
		return teff,e1,e2
	if law == 'squareroot':
		teff,s1,s2 = np.loadtxt('lds_tables/'+table_name,unpack=True,usecols=(3,21,22))
		return teff,s1,s2

	print 'Limb-darkening law '+law+' not supported.' 
	sys.exit()


