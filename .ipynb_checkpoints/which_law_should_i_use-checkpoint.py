#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing as mp
import os
import pyfits
import lmfit
import sys

import Utils

##################### CUSTOMIZABLE OPTIONS ###########################

# Define ld laws to simulate transits from:
ld_laws = ['linear','quadratic','logarithmic','squareroot','three-param']

# Define number of in-transit points in each transit simulation:
N = 100
# Number of transits to simulate:
n_try = 300

# Precisions (in ppm) of the lightcurves to be simulated (can be just one):
precisions = [10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,200.,300.,400.,500.,\
              600.,700.,800.,900.,1000.,2000.,3000.]

# Define transit parameters for the simulations:
P = 4.234516 #days. Presumably ld-exosim takes the input in days?
b = 0.303
a = 13.06 # 13.06±0.83 a/R* semi-major axis to stellar radius ratio, from Hartman 2010
p = 0.0737 # planet-to-star radius ratio

# Define stellar parameters for the simulations:
Teff = 5079.   # Temperature in K, +/- 88
logg = 4.56     # Log-gravity (dex/cgs)
feh = -0.04    # Metallicity # From Hartman 2010, but other papers give different values. does the ld-exosim rec change??
vturb = 2.0    # Microturbulent velocity (km/s)

# Finally, select the limb-darkening table to use (default is the Kepler+ATLAS one,
# but you can generate your own from here: https://github.com/nespinoza/limb-darkening):
ld_table_name = 'kepler_atlas_lds.dat'

##################### GET LDS FROM TABLES ############################
print('\n ')
print('\t    Running "What LD law should I use?" v.1.0.')
print('\t    Author: Nestor Espinoza (nsespino@uc.cl)')

# First, get LDs for each law:
lds = {}
for ld_law in (ld_laws + ['non-linear']):
    lds[ld_law] = Utils.read_ld_table(law = ld_law, input_teff = Teff, input_logg = logg, \
                                      input_feh = feh, input_vturb = vturb, table_name = ld_table_name)

 ##################### PREPARE OUTPUT FOLDERS #########################

if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists('results/sic'):
    os.mkdir('results/sic')

# Calculate the inclination:
inclination = np.arccos(b/a)*180./np.pi

t0 = 0.0
for precision in precisions:
    output_folder = 'results/sic/N_'+str(N)+'_precision_'+str(precision)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    else:
        continue
    ##################### SIMULATION AND ANALYSIS ########################

    pfloat = {}
    afloat = {}
    ifloat = {}
    pfloat_noisy = {}
    afloat_noisy = {}
    ifloat_noisy = {}
    for ld_law in ld_laws:
        pfloat[ld_law] = []
        afloat[ld_law] = []
        ifloat[ld_law] = []
        pfloat_noisy[ld_law] = []
        afloat_noisy[ld_law] = []
        ifloat_noisy[ld_law] = []

    for i in range(n_try):
        result = []
        # Calculate duration of the transit:
        transit_time = Utils.get_transit_duration(P, p, 1./a, np.arccos(b/a))
        # Generate times based on this duration:
        times = np.linspace(-(transit_time)/2.0,(transit_time)/2.0,N)
        # Add two hundred points before and after transit, just to have some points off-transit:
        delta_times = np.diff(times)[0]
        time_points_before = times[0]-(np.arange(1,201,1)*delta_times)
        time_points_after = times[-1]+(np.arange(1,201,1)*delta_times)
        times = np.append( time_points_before ,times )
        times = np.append( times, time_points_after )
        # Generate random normal noise:
        noise = np.random.normal(0,precision*1e-6,len(times))
        # Save noise:
        pyfits.PrimaryHDU(noise).writeto(output_folder+'/noise_ntry_'+str(i)+'.fits') 
        # Generate random time offset:
        time_offset = np.random.uniform(-delta_times,delta_times)
        t = np.copy(times) + time_offset
        # Save the times:
        pyfits.PrimaryHDU(t).writeto(output_folder+'/times_ntry_'+str(i)+'.fits')
        # Now, generate transit lightcurve using the coefficients c1,c2,c3,c4 from models and the input parameters:
        params,m = Utils.init_batman(t,P,inclination,a,p,t0,lds['non-linear'],ld_law = 'non-linear')
        transit = m.light_curve(params)
        # Save the transit:
        pyfits.PrimaryHDU(transit).writeto(output_folder+'/transit_ntry_'+str(i)+'.fits')
        for ld_law in ld_laws:
                # Fit with free LD coefficients:
                try:
                    if ld_law == 'linear':
                        p_lsq2, coeff1_lsq2, i_lsq2, a_lsq2 = Utils.fit_transit_floating_lds(t, transit, p, lds[ld_law][0], \
                                                                                                 None, inclination, a, P, t0, ld_law)
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2],ld_law = ld_law)
                    elif ld_law == 'three-param':
                        p_lsq2, coeff1_lsq2, coeff2_lsq2, coeff3_lsq2, i_lsq2, a_lsq2 = Utils.fit_transit_floating_lds(t, transit, p, lds[ld_law][0], \
                                                                                                                           lds[ld_law][1], inclination, a, P, \
                                                                                                                           t0, ld_law, guess_coeff3 = lds[ld_law][2])
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2,coeff2_lsq2,coeff3_lsq2],ld_law = ld_law)
                    else:
                        p_lsq2, coeff1_lsq2, coeff2_lsq2, i_lsq2, a_lsq2 = Utils.fit_transit_floating_lds(t, transit, p, lds[ld_law][0], \
                                                                                                              lds[ld_law][1], inclination, a, P, t0, ld_law)
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2,coeff2_lsq2],ld_law = ld_law)
                except:
                    print('Fit failed for bias calculation of LD law '+ld_law+' iteration '+str(i)+' of '+str(n_try)+'.')
                    if ld_law == 'linear':
                        p_lsq2, coeff1_lsq2, i_lsq2, a_lsq2 = p, lds[ld_law][0], inclination, a
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2],ld_law = ld_law)
                    elif ld_law == 'three-param':
                        p_lsq2, coeff1_lsq2, coeff2_lsq2, coeff3_lsq2, i_lsq2, a_lsq2 = p, lds[ld_law][0], lds[ld_law][1], lds[ld_law][2], inclination, a
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2,coeff2_lsq2,coeff3_lsq2],ld_law = ld_law)
                    else:
                        p_lsq2, coeff1_lsq2, coeff2_lsq2, i_lsq2, a_lsq2 = p, lds[ld_law][0], lds[ld_law][1], inclination, a
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2,coeff2_lsq2],ld_law = ld_law)                        
                             
                pfloat[ld_law].append(np.copy(p_lsq2))
                afloat[ld_law].append(np.copy(a_lsq2))
                ifloat[ld_law].append(np.copy(i_lsq2))

                # Save best-fit transit with floating LDs:
                best_fit_transit_float = m_lsq2.light_curve(params_lsq2)
                pyfits.PrimaryHDU(best_fit_transit_float).writeto(output_folder+'/best_fit_float_noiseless__'+ld_law+'_ntry_'+str(i)+'.fits')

                # Same thing, now for noisy LC:
                try:
                    if ld_law == 'linear':
                        p_lsq2, coeff1_lsq2, i_lsq2, a_lsq2 = Utils.fit_transit_floating_lds(t, transit+noise, p, lds[ld_law][0], \
                                                                                     None, inclination, a, P, t0, ld_law)
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2],ld_law = ld_law)
                    elif ld_law == 'three-param':
                        p_lsq2, coeff1_lsq2, coeff2_lsq2, coeff3_lsq2, i_lsq2, a_lsq2 = Utils.fit_transit_floating_lds(t, transit+noise, p, lds[ld_law][0], \
                                                                                        lds[ld_law][1], inclination, a, P, t0, ld_law, guess_coeff3 = lds[ld_law][2])
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2,coeff2_lsq2,coeff3_lsq2],ld_law = ld_law)
                    else:
                        p_lsq2, coeff1_lsq2, coeff2_lsq2, i_lsq2, a_lsq2 = Utils.fit_transit_floating_lds(t, transit+noise, p, lds[ld_law][0], \
                                                                                                  lds[ld_law][1], inclination, a, P, t0, ld_law)
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2,coeff2_lsq2],ld_law = ld_law)
                except:
                    print('Fit failed for precision calculation of LD law '+ld_law+' iteration '+str(i)+' of '+str(n_try)+'.')
                    if ld_law == 'linear':
                        p_lsq2, coeff1_lsq2, i_lsq2, a_lsq2 = p, lds[ld_law][0], inclination, a
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2],ld_law = ld_law)
                    elif ld_law == 'three-param':
                        p_lsq2, coeff1_lsq2, coeff2_lsq2, coeff3_lsq2, i_lsq2, a_lsq2 = p, lds[ld_law][0], lds[ld_law][1], lds[ld_law][2], inclination, a
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2,coeff2_lsq2,coeff3_lsq2],ld_law = ld_law)
                    else:
                        p_lsq2, coeff1_lsq2, coeff2_lsq2, i_lsq2, a_lsq2 = p, lds[ld_law][0], lds[ld_law][1], inclination, a
                        params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2,coeff2_lsq2],ld_law = ld_law)
                pfloat_noisy[ld_law].append(np.copy(p_lsq2))
                afloat_noisy[ld_law].append(np.copy(a_lsq2))
                ifloat_noisy[ld_law].append(np.copy(i_lsq2))

                # Save best-fit transit with floating LDs:
                best_fit_transit_float = m_lsq2.light_curve(params_lsq2)
                pyfits.PrimaryHDU(best_fit_transit_float).writeto(output_folder+'/best_fit_float_noisy__'+ld_law+'_ntry_'+str(i)+'.fits')

    print('\t ######################################################')
    print('\t Simulations finished with precision of '+str(precision)+' ppm,')
    print('\t and '+str(N)+' in-transit points.')
    print('\t ######################################################')
    for ld_law in ld_laws:
        fout = open(output_folder+'/results_'+ld_law+'.dat','w')
        fout.write('# Bias on p \t Precision on p \t Bias on a \t Precision on a \t Bias on i \t Precision on i\n')
        print('\t Results for '+ld_law+' LD law (free LD coeffs):')
        print('\t ------------------------------')
        bias_float_p = np.median(pfloat[ld_law])-p
        precision_float_p = np.sqrt(np.var(pfloat_noisy[ld_law]))
        bias_float_a = np.median(afloat[ld_law])-a
        precision_float_a = np.sqrt(np.var(afloat_noisy[ld_law]))
        bias_float_i = np.median(ifloat[ld_law])-inclination
        precision_float_i = np.sqrt(np.var(ifloat_noisy[ld_law]))
        fout.write(str(bias_float_p)+'\t'+str(precision_float_p)+'\t'+\
                        str(bias_float_a)+'\t'+str(precision_float_a)+'\t'+ 
                        str(bias_float_i)+'\t'+str(precision_float_i)+'\n')
        print('\t Planet-to-star radius ratio (p = Rp/Rs):')    
        print('\t Bias:                ',bias_float_p)
        print('\t sqrt(Variance):      ',precision_float_p)
        print('\t Bias/sqrt(Variance): ',np.abs(bias_float_p/precision_float_p))
        print('\t MSE:                 ',(bias_float_p**2 + precision_float_p**2))
        print('')
        print('\t Scaled semi-major axis (a=a/Rs):')
        print('\t Bias:                ',bias_float_a)
        print('\t sqrt(Variance):      ',precision_float_a)
        print('\t Bias/sqrt(Variance): ',np.abs(bias_float_a/precision_float_a))
        print('\t MSE:                 ',bias_float_a**2+precision_float_a**2)
        print('')
        print('\t Inclination (degrees, i):')
        print('\t Bias:                ',bias_float_i)
        print('\t sqrt(Variance):      ',precision_float_i)
        print('\t Bias/sqrt(Variance): ',np.abs(bias_float_i/precision_float_i))
        print('\t MSE:                 ',bias_float_i**2 + precision_float_i**2)
        print('\t ------------------------------')
        print('')
    fout.close()

print('\t    Done!')
print('\n')
