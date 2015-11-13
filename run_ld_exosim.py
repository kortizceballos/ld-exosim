import numpy as np
import multiprocessing as mp
import os
import pymc
import pyfits
import lmfit
import sys

sys.path.append('../utilities/')
import Utils

##################### CUSTOMIZABLE OPTIONS ###########################

# Define ld_law to simulate transits from:
ld_law = 'three-param'

# Define constant values on the simulation (i.e., period, P, time of transit 
# center, t0, and impact parameter, b, number of points in each transit of 
# the simulation, N, and number of transit to simulate per grid point, n_try:
P = 1.0
t0 = 0.0
b = 0.0
N = 1000
n_try = 100

# Define number of cores to use:
ncores = 5

# Define the grid to explore; first, scaled semi-major axes (same as in 
# Espinoza & Jordan, 2015):
sim_a = [3.27, 3.92, 4.87, 6.45, 9.52, 18.18, 200]

# Now planet-to-star radius ratios; from 0.01 to 0.21 in 0.02 steps:
sim_p = [0.01,0.07,0.13]

# Finally, select the limb-darkening table to use (default is the Kepler+ATLAS one):
ld_table_name = 'kepler_atlas_lds.dat'

##################### GET LDS FROM TABLES ############################

# First, get LDs for non-linear law:
teffs, c1, c2, c3, c4 = Utils.read_ld_table(law = 'non-linear', table_name = ld_table_name)

# Now, get LDs for the selected LD law:
if ld_law == 'three-param':
	teffs, coeff1, coeff2, coeff3 = Utils.read_ld_table(law = ld_law, table_name = ld_table_name)
else:
	teffs, coeff1, coeff2 = Utils.read_ld_table(law = ld_law, table_name = ld_table_name)

##################### PREPARE OUTPUT FOLDERS #########################

if not os.path.exists('results'):
	os.mkdir('results')

output_folder = 'results/'+ld_law+'_b_'+str(b)
if not os.path.exists(output_folder):
        os.mkdir(output_folder)

##################### SIMULATION AND ANALYSIS ########################

# Save grid values of p and a into a list (easier for multi-processing):
grid_values = []
counter = 0
for a in sim_a:
	for p in sim_p:
		grid_values.append([a,p])
		# Create folder for the outputs of this grid:
		os.mkdir(output_folder+'/grid_files_'+str(counter))
		counter = counter + 1

def get_sigma_mad(x):
    mad = np.median(np.abs(x-np.median(x)))
    return 1.4826*mad

def run_simulations(counter):
    result = []
    # Take the values of a and p, calculate inclination:
    a,p = grid_values[counter]
    inclination = np.arccos(b/a)*180./np.pi
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
    for j in range(len(teffs)):
	    pfixed = []
	    pfloat = []
	    afixed = []
	    afloat = []
	    ifixed = []
	    ifloat = []  
	    p_file = open(output_folder+'/grid_files_'+str(counter)+'/p_vals_teff_'+str(teffs[j])+'.dat','w')
	    a_file = open(output_folder+'/grid_files_'+str(counter)+'/a_vals_teff_'+str(teffs[j])+'.dat','w')
	    i_file = open(output_folder+'/grid_files_'+str(counter)+'/i_vals_teff_'+str(teffs[j])+'.dat','w')
	    ld_coeffs_file = open(output_folder+'/grid_files_'+str(counter)+'/ld_coeffs_teff_'+str(teffs[j])+'.dat','w')
            if ld_law == 'three-param':
		ld_coeffs_file.write('# coeff1_fitted \t coeff2_fitted \t coeff3_fitted \t coeff1_fixed \t coeff2_fixed \t coeff3_fixed \n')
	    else:
	    	ld_coeffs_file.write('# coeff1_fitted \t coeff2_fitted \t coeff1_fixed \t coeff2_fixed\n')
	    p_file.write('# p_fit_fixed_lds \t p_fit_floating_lds \n')
	    a_file.write('# a_fit_fixed_lds \t a_fit_floating_lds \n')
	    i_file.write('# i_fit_fixed_lds \t i_fit_floating_lds \n')
	    for i in range(n_try):
		# Generate random time offset:
		time_offset = np.random.uniform(-delta_times,delta_times)
		t = np.copy(times) + time_offset
		# Save the times:
		pyfits.PrimaryHDU(t).writeto(output_folder+'/grid_files_'+str(counter)+\
				     '/times_teff_'+str(teffs[j])+'_ntry_'+str(i)+'.fits')
		# Now, generate transit lightcurve using the coefficients c1,c2,c3,c4 from models and the input parameters:
		params,m = Utils.init_batman(t,P,inclination,a,p,t0,[c1[j],c2[j], c3[j], c4[j]],ld_law = 'non-linear')
		transit = m.light_curve(params)
		# Save the transit:
		pyfits.PrimaryHDU(transit).writeto(output_folder+'/grid_files_'+str(counter)+\
				  '/transit_teff_'+str(teffs[j])+'_ntry_'+str(i)+'.fits')    
		# Fit it using fixed two-parameter limb-darkening coefficients:
		if ld_law == 'three-param':
			p_lsq, i_lsq, a_lsq = Utils.fit_transit_fixed_lds(t, transit, p, coeff1[j], coeff2[j], inclination, a, P, t0, ld_law, coeff3 = coeff3[j])
			params_lsq,m_lsq = Utils.init_batman(t,P,i_lsq,a_lsq,p_lsq,t0,[coeff1[j],coeff2[j],coeff3[j]],ld_law = ld_law) 
		else:
			p_lsq, i_lsq, a_lsq = Utils.fit_transit_fixed_lds(t, transit, p, coeff1[j], coeff2[j], inclination, a, P, t0, ld_law)
			params_lsq,m_lsq = Utils.init_batman(t,P,i_lsq,a_lsq,p_lsq,t0,[coeff1[j],coeff2[j]],ld_law = ld_law)
		# Save the fitted parameters:
		pfixed.append(np.copy(p_lsq))
		afixed.append(np.copy(a_lsq))
		ifixed.append(np.copy(i_lsq))
		# Save best-fit transit with fixed LDs:
		best_fit_transit_fixed = m_lsq.light_curve(params_lsq)
		pyfits.PrimaryHDU(best_fit_transit_fixed).writeto(output_folder+'/grid_files_'+str(counter)+\
				  '/best_fit_fixed_transit_teff_'+str(teffs[j])+'_ntry_'+str(i)+'.fits')

		# Now fit with free LD coefficients:
		if ld_law == 'three-param':
                        p_lsq2, coeff1_lsq2, coeff2_lsq2, coeff3_lsq2, i_lsq2, a_lsq2 = Utils.fit_transit_floating_lds(t, transit, p, coeff1[j], \
                                                                                        coeff2[j], inclination, a, P, t0, ld_law, guess_coeff3 = coeff3[j])
			params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2,coeff2_lsq2,coeff3_lsq2],ld_law = ld_law)
			ld_coeffs_file.write(str(coeff1_lsq2)+'\t'+str(coeff2_lsq2)+'\t'+str(coeff3_lsq2)+'\t'+str(coeff1[j])+'\t'+str(coeff2[j])+'\t'+str(coeff3[j])+'\n')
		else:
			p_lsq2, coeff1_lsq2, coeff2_lsq2, i_lsq2, a_lsq2 = Utils.fit_transit_floating_lds(t, transit, p, coeff1[j], \
										  	coeff2[j], inclination, a, P, t0, ld_law)
			params_lsq2,m_lsq2 = Utils.init_batman(t,P,i_lsq2,a_lsq2,p_lsq2,t0,[coeff1_lsq2,coeff2_lsq2],ld_law = ld_law)
			ld_coeffs_file.write(str(coeff1_lsq2)+'\t'+str(coeff2_lsq2)+'\t'+str(coeff1[j])+'\t'+str(coeff2[j])+'\n')
		pfloat.append(np.copy(p_lsq2))
		afloat.append(np.copy(a_lsq2))
		ifloat.append(np.copy(i_lsq2))

		# Save best-fit transit with floating LDs:
		best_fit_transit_float = m_lsq2.light_curve(params_lsq2)
		pyfits.PrimaryHDU(best_fit_transit_float).writeto(output_folder+'/grid_files_'+str(counter)+\
				  '/best_fit_floating_transit_teff_'+str(teffs[j])+'_ntry_'+str(i)+'.fits')
		p_file.write(str(p_lsq)+'\t'+str(p_lsq2)+'\n')
		a_file.write(str(a_lsq)+'\t'+str(a_lsq2)+'\n')
		i_file.write(str(i_lsq)+'\t'+str(i_lsq2)+'\n')
	    p_file.close()
	    a_file.close()
	    i_file.close()
	    p_fixed = np.median(pfixed)
	    a_fixed = np.median(afixed)
	    i_fixed = np.median(ifixed)
	    sigma_p_fixed = get_sigma_mad(pfixed)
	    sigma_a_fixed = get_sigma_mad(afixed)
	    sigma_i_fixed = get_sigma_mad(ifixed)
	    p_float = np.median(pfloat)
	    a_float = np.median(afloat)
	    i_float = np.median(ifloat)
	    sigma_p_float = get_sigma_mad(pfloat)
	    sigma_a_float = get_sigma_mad(afloat)
	    sigma_i_float = get_sigma_mad(ifloat)
	    result.append([teffs[j],p_fixed,sigma_p_fixed,p_float,sigma_p_float,a_fixed,sigma_a_fixed,a_float,sigma_a_float,\
		    i_fixed,sigma_i_fixed,i_float,sigma_i_float])
    return result

# Run simulations on all the grids with multi-processing:
pool = mp.Pool(processes=ncores)
results = pool.map(run_simulations, range(len(grid_values)))
pool.terminate()

# Save final results in human-readable form:
output_file = open(output_folder+'/final_results.dat','w')
output_file.write('# Results of the simulations. Done for the '+ld_law+' LD law with:\n')
output_file.write('# Period = '+str(P)+', t0 = '+str(t0)+', b = '+str(b)+', N = '+str(N)+\
                  ' and n_try = '+str(n_try)+'\n')

output_file.write('#\n# gnum \t input_p \t input_a \t input_teff \t p_fixed \t sigma_p_fixed '+\
                  '\t p_float \t sigma_p_float \t a_fixed \t sigma_a_fixed \t a_float \t sigma_a_float'+\
                  '\t i_fixed \t sigma_i_fixed \t i_float \t sigma_i_float\n')

for i in range(len(grid_values)):
        a,p = grid_values[i]
        common_output_string = str(i)+'\t'+str(p)+'\t'+str(a)
        c_results = results[i]
        for j in range(len(c_results)):
              output_string = common_output_string
              t_c_results = c_results[j]
              for k in range(len(t_c_results)):
                  output_string = output_string+'\t'+str(t_c_results[k])
              output_file.write(output_string+'\n')

output_file.close()
print 'Done!'
