from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import correlate
from astropy.table import Table
from SpectralReduction import *
import time


#EDIT THESE LINES #############
##Directory where all the data is stored 
Carmenes_Aframes = glob.glob('Data/HD209458_Carmenes/*nir_A.fits')

#file name of the model atmosphere 
modelAtm = ascii.read('/Users/aurora/Dropbox/MetalHydrideTests/petitRADTRANS-master/modelAtm_HD209_H2O.dat')

starName = 'HD209'

spectralRes = 80000 #spectral resolution (CARMENES NIR = 80000)

waveRange = [10000, 12000] # in angstroms

planetRotVel = 2.0

#orbital parameters of the system
To = 2452826.629283 #Bonomo+2017
Period = 3.52474859 #Stassun+2017
Kp = 151.0 #km/s value from Sanchez-Lopez+2019
vsys = -14.7652 #Mazeh+2000

StellarRad = 1.155 #Torres+2008

#Indices when transit starts and ends 
transitIndices = [15, 61]

#cut spectra -- use this to get rid of spectra that are too noisy
#the indices should refer to the spectra after they have been sorted in the table
cutSpecs = [0, 1, 2, 3, 4, 5, 6, 7, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]

#set to True if you want to inject the model into the data for testing purposes
inject = False

############################## DONE w/ EDITING PARAMETERS ###############

#script
start_time = time.time()

#open the model atmosphere file and extract the data
modelWave = modelAtm['Wavelength']*10**4
modelWave_broad, modelRad_broad = prepareModel(modelWave, modelAtm['Radius'], SpectralRes, planetRotVel, waveRange)
modelTrans_broad = 1 - ((modelRad_broad * 6.9911*10**7)**2 / (StellarRad * 6.9634*10**8)**2 )
modelTrans = 1 - ((modelAtm['Radius'] * 6.9911*10**7)**2 / (StellarRad * 6.9634*10**8)**2 )

Fluxes = []
Sigmas = []
Waves = []
restWaves = []
obsTimes = []
obsTimesUT = []
phases = []
rvs = []
MJDs = []
vbar = []

#open fits files and extract the necessary information 
for i in range(len(Carmenes_Aframes)):
    
    with fits.open(Carmenes_Aframes[i], memmap=False) as f:
        
        flux1 = f[1].data[0][8:] #nans at the beginning
        wave1 = f[4].data[0][8:]
        sigma1 = f[3].data[0][8:]
        flux2 = f[1].data[1] #less order overlap
        wave2 = f[4].data[1]
        sigma2 = f[3].data[1]
        flux3 = f[1].data[2]
        wave3 = f[4].data[2]
        sigma3 = f[3].data[2]
        flux4 = f[1].data[3]
        wave4 = f[4].data[3]
        sigma4 = f[3].data[3]
        obsTime = f[0].header['DATE-OBS'][11:]
        BJD = f[0].header['HIERARCH CARACAL BJD'] + 2400000
        vbar = f[0].header['HIERARCH CARACAL BERV']
        UT = f[0].header['UT']
        MJD = f[0].header['MJD-OBS']+2400000.5
        airmass = f[0].header['AIRMASS']
        snr = (f[0].header['HIERARCH CARACAL FOX SNR 0'] + f[0].header['HIERARCH CARACAL FOX SNR 1'] + f[0].header['HIERARCH CARACAL FOX SNR 2'])/ 3
    
    #combine the fluxes and wavelengths into a single array
    fluxes = np.concatenate((flux1, flux2, flux3, flux4), axis = None) 
    waves = np.concatenate((wave1, wave2, wave3, wave4), axis=None)
    sigmas = np.concatenate((sigma1, sigma2, sigma3, sigma4), axis = None)
    #order by wavelength
    t_init = Table( [waves, fluxes, sigmas], names=['waves', 'fluxes', 'sigmas'])
    t_init.sort('waves')
    
    SNRs.append(snr)
    obsTimes.append(obsTime)
    phase = calculateOrbitalPhase(BJD, To, Period)
    airmasses.append(airmass)

    #correct for the barycentric velocity etc.
    vbar.append(vbar)
    restWave = waves / (((-vbar + vsys)/(2.998*10**5)) + 1)
    restWaves.append(restWave)

    Fluxes.append(t_init['fluxes'])
    Waves.append(t_init['waves'])
    Sigmas.append(t_init['sigmas'])
    obsTimesUT.append(UT)
    
    rv = calculatePlanetRV(Kp, phase)
    rvs.append(rv) 
    phases.append(phase)
    MJDs.append(MJD)
    BJDs.append(BJD)
    

#order them correctly by time
Flux_array = np.asarray(Fluxes)
Wave_array = np.asarray(restWaves)
Sigma_array = np.asarray(Sigmas)

#order them correctly by time
t = Table( [Flux_array, Wave_array, Sigma_array, obsTimes, obsTimesUT, phases, rvs, vbars, MJDs], names=['fluxes', 'waves', 'sigmas', 'times', 'UTs', 'phases', 'rvs', 'vbars', 'MJDs'])
t.sort('MJDs')
t.reverse()

t.remove_rows(cutSpecs)

#look at the spectra
colors = cm.RdPu_r(np.linspace(0,1,len(t)))
for j in range(len(t)):
    plt.plot(t['waves'][j], t['fluxes'][j], color = colors[j])
plt.show()

#inject a signal
flux_wSignal = []
for i in range(len(t)):
    if i < transitIndices[0] or i > transitIndices[1]:
        fluxes_inject = t['fluxes'][i]
    else: 
        fluxes_inject = injectPlanetSignal(t['waves'][i], t['fluxes'][i], modelWave_broad, modelTrans_broad, t['rvs'][i])
    flux_wSignal.append(fluxes_inject)
flux_wSignal = np.asarray(flux_wSignal)


#do the reduction
if inject == True: 
    rv_grid, cc_grid = doTheReduction(t['waves'], flux_wSignal, modelWave_broad, modelTrans_broad, 8, t['phases'], plots=False)
else:
    rv_grid, cc_grid = doTheReduction(t['waves'], t['fluxes'], modelWave_broad, modelTrans_broad, 12, t['phases'], plots=False)

#shift the cross correlations into the planet's rest frame
new_rv_grid, rest_ccs = shiftXcorl(rv_grid, cc_grid, t['rvs'])

#flatten the grid in time so that we get a signal at the planet's rest frame
flat_cc = flattenXcorl(rest_ccs[transitIndices[0]:transitIndices[1]])

print("The code took %s seconds to run" % (time.time() - start_time))

#plot to assess how it went! 
fig, ax = plt.subplots()
im = ax.imshow(cc_grid, cmap=cm.gray, extent=[np.min(rv_grid), np.max(rv_grid), 0, len(cc_grid)])
ax.set_xlabel('Radial Velocity (km/s)')
ax.set_ylabel('Spectrum Number')
ax.set_aspect(2)
plt.plot([-50,50], [len(t) - transitIndices[0], len(t) - transitIndices[0]], 'r--')
plt.plot([-50,50], [len(t) - transitIndices[1], len(t) - transitIndices[1]], 'r--')
plt.plot(t['rvs'][transitIndices[1]:], np.arange((len(t) - transitIndices[1]),0,-1), 'k--')
plt.plot(t['rvs'][0:transitIndices[0]], np.arange(len(t),(len(t) - transitIndices[0]),-1), 'k--') 
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(rest_ccs, cmap=cm.gray, extent=[np.min(new_rv_grid), np.max(new_rv_grid), 0, len(t)])
ax.set_xlabel('Radial Velocity (km/s)')
ax.set_ylabel('Spectrum Number')
plt.plot([-50,50], [len(t) - transitIndices[0], len(t) - transitIndices[0]], 'r--')
plt.plot([-50,50], [len(t) - transitIndices[1], len(t) - transitIndices[1]], 'r--')
plt.show()


plt.plot(new_rv_grid, flat_cc)
plt.xlabel('Radial Velocity')
plt.ylabel('Cross Correlation')
plt.show()

SNR_injected = calculateSNR(new_rv_grid, flat_cc)
print('The SNR of the injected signal is '+ str(SNR_injected))
pdb.set_trace()

best_Kp, best_v = findKpVsys(rv_grid, cc_grid[transitIndices[0]:transitIndices[1]], Kp, t['phases'][transitIndices[0]:transitIndices[1]])

print('The best Kp and Vsys correction are ' + str(best_Kp) + ' ' + str(best_v))

pdb.set_trace()
