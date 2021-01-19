from astropy.io import fits, ascii
import matplotlib.pyplot as plt
import numpy as np
import glob
import pdb
import matplotlib.cm as cm
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import correlate
from astropy.table import Table
from PyAstronomy import pyasl
from scipy import signal
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.ndimage import gaussian_filter1d

####################
#Simple removal of the host star's spectrum by simply dividing by the
#average of all the fluxes in time (since host star is stationary)
#AframeFluxes is an array with all the time series spectra in position 1 and all the pixels in 2
####################
def removeHostStar(AframeFluxes):

    Aframes_noStar = np.zeros(AframeFluxes.shape)
    meanAframes = np.nanmedian(AframeFluxes, axis=0)
    for i in range(len(AframeFluxes)):
        #mean of all the time series out of transit
        #Fout_list = np.concatenate((AframeFluxes[:transitIndices[0]], AframeFluxes[transitIndices[1]:]) )
        #divide each entry by the mean (all the wavelength arrays are the same for each order) 
        Aframes_noStar[i] = AframeFluxes[i]/meanAframes
        
    
    return Aframes_noStar

####################
#instead divide by only average spectra that are not taken during transit
#Need to use this function if the element/molecule is in the stellar spectrum as well (will cause doppler shadow)
####################
def removeHostStarOutTransit(AframeFluxes, indices):
    Aframes_noStar = np.zeros(AframeFluxes.shape)
    out_of_transit = np.concatenate((AframeFluxes[0:indices[0]] , AframeFluxes[indices[1]:]), axis=0)
    meanAframes = np.nanmedian(out_of_transit, axis=0)
    for j in range(len(AframeFluxes)):
        #mean of all the time series out of transit
        #Fout_list = np.concatenate((AframeFluxes[:transitIndices[0]], AframeFluxes[transitIndices[1]:]) )
        #divide each entry by the mean (all the wavelength arrays are the same for each order) 
        Aframes_noStar[j] = AframeFluxes[j]/meanAframes
        
    
    return Aframes_noStar

####################
#Function to weight each pixel by the standard deviation in time - should downweight noisy edge pixels or pixels with telluric residuals, etc. 
#Good to increase the signal, but don't use if trying to get out physical parameters like residual amplitude/ Radius of planet
####################
def divideByStd(AframeFluxes):

    newFluxes = np.zeros(AframeFluxes.shape)
    for i in range(len(AframeFluxes[0])):
        #mean of all the time series out of transit

        std = np.std(AframeFluxes[:, i])
        newFluxes[:,i] = ((AframeFluxes[:,i] - 1)/(std))+1
    
    return newFluxes

####################
#Function to weight the spectra by their signal to noise ratio so that the noisiest spectra don't contribute the most
#Good to increase the signal, but don't use if trying to get out physical parameters like residual amplitude/ Radius of planet
####################
def weightBySNR(AframeFluxes):

    weightedFluxes = np.zeros(AframeFluxes.shape)
    for i in range(len(AframeFluxes)):
        std = np.nanstd(AframeFluxes[i])
        weightedFlux = ((AframeFluxes[i] -1)/ (std))+1
        weightedFluxes[i] = weightedFlux 

    return weightedFluxes

####################
#Sigma clipping in the time direction. Now set to 5 sigma clipping - but could go more stringent to 4 sigma
####################
def removeVertOutliers(AframeFluxes):

    for i in range(len(AframeFluxes[0])):

        #mean of all the time series
        meanFlux = np.nanmedian(AframeFluxes[:,i])
        stdFlux = np.std(AframeFluxes[:,i])
        distMean = np.abs(AframeFluxes[:,i] - meanFlux)
        badSigmas = np.where(distMean > 5*stdFlux)
        clippedFluxes = AframeFluxes 
        if len(badSigmas[0]) > 0:
            #make the outlier the mean value
            clippedFluxes[badSigmas[0][0], i] = np.nan

    for k in range(len(clippedFluxes)):
        nans, x = nan_helper(clippedFluxes[k])
        clippedFluxes[k][nans] = np.interp(x(nans), x(~nans), clippedFluxes[k][~nans])


    
    return clippedFluxes

def normalize(flux):
    
    normFlux = np.zeros(flux.shape)
    for i in range(len(flux)):
        meanF = np.nanmean(flux[i])
        normF = flux[i]/meanF
        normFlux[i] = normF
    
    return normFlux

def normalizeSig(flux, sigmas):
    
    normSig = np.zeros(flux.shape)
    for i in range(len(flux)):
        meanF = np.nanmean(flux[i])
        normS = sigmas[i]/meanF
        normSig[i] = normS
    
    return normSig

####################
#remove 5 sigma outliers but in the wavelength direction this time
#tried a few different ways to do this
####################
def remove5sOutliers(AframeFluxes, AframeWaves, AframeSigmas):

    for i in range(len(AframeFluxes)):

        #check if there are any nans and interpolate over them
        nans, x = nan_helper(AframeFluxes[i])
        AframeFluxes[i][nans] = np.interp(x(nans), x(~nans), AframeFluxes[i][~nans])
        AframeSigmas[i][nans] = 5*np.interp(x(nans), x(~nans), AframeFluxes[i][~nans]) #make the sigmas large where nans

        #std of each spectrum 
        stdAframe = np.nanstd(AframeFluxes[i])
        medianAframe = np.nanmedian(AframeFluxes[i])
        #find points where the flux is more than 5 sigma away from the mean
        distMean = np.abs(AframeFluxes[i] - medianAframe)
        sigmaPoints = np.where( distMean > 3*stdAframe)

        correctedFluxes = AframeFluxes
        if len(sigmaPoints) > 0: 
            correctedFluxes[i][sigmaPoints] = 1.0
        
    return correctedFluxes

####################
#remove any residual shape to the spectral orders
####################
def removeBlazeFunction(wave, flux):


    if len(flux.shape) > 1: 
        correctedFlux = np.zeros(flux.shape)
        for i in range(len(flux)):
            #try a very smoothed spline
            #if the order has that deep absorption band mask it (need to treat this more generally later)
            weights_mask = np.ones(len(wave))
            #if wave[0][j][0] < 10000 and wave[0][j][0] > 9900:
            #    weights_mask[2100:3300] = 0.1
            poly = np.polyfit(wave, flux[i], 3)
            p = np.poly1d(poly)
            #spl = UnivariateSpline(wave, flux[i], w = weights_mask, s=1)
            #plt.plot(wave, flux[i])
            #plt.plot(wave, spl(wave))
            #plt.show()
            #correctedFlux[i] = flux[i]/spl(wave)
            correctedFlux[i] = flux[i]/p(wave)
            #plt.plot(wave, correctedFlux[i])
            #plt.show()
            #pdb.set_trace()

    elif len(flux.shape) == 1:
        poly = np.polyfit(wave, flux, 3)
        #spl = UnivariateSpline(wave, flux, s=1)
        p = np.poly1d(poly)
        correctedFlux = flux/p(wave)
        #correctedFlux = flux/spl(wave)
        #plt.plot(wave, flux, wave, correctedFlux)
        #plt.show()
        #pdb.set_trace()

    return correctedFlux

def crossCorrelate(wavelength, flux, waveTemplate, fluxTemplate):

    #put the template on the same wavegrid and
    tempSpl = interp1d(waveTemplate, fluxTemplate)
    fluxTemp = tempSpl(wavelength)
    #remove any linear trend
    fluxTemp = removeBlazeFunction(wavelength, fluxTemp)
    flatFlux = removeBlazeFunction(wavelength, flux)

    fluxTemp[0:1500] = 1.0 #get rid of any lines within the points on the edge since those will be shifted in and out of view and cause vertical structure
    fluxTemp[-1500:] = 1.0
    paddedFluxTemp = np.concatenate((np.ones(2000), fluxTemp, np.ones(2000))) #add padding of 500 on each end
    #figure out the wavegrid
    logUpperWaveGrid = (np.zeros(2000) + np.log10(wavelength[-1])) +  (np.arange(1,2001,1)*(0.2*np.log10(np.e)/(2.998*10**5)))
    logLowerWaveGrid = (np.zeros(2000) + np.log10(wavelength[0])) -  (np.arange(2001,1,-1)*(0.2*np.log10(np.e)/(2.998*10**5)))
    paddedWaveTemp = np.concatenate( (10**logLowerWaveGrid, wavelength, 10**logUpperWaveGrid))
    
    ccGrid = []
    if len(flux.shape) > 1: 
        for i in range(len(flux)):
            cc = np.correlate(paddedFluxTemp - np.mean(paddedFluxTemp), flatFlux[i] - np.mean(flatFlux[i]), mode = 'valid')
            cc_norm = cc / (len(flatFlux[i]) * np.std(flatFlux[i]) * np.std(paddedFluxTemp)) #todcor def
            #cc_norm = cc / (np.std(flatFlux[i]) * np.std(paddedFluxTemp)) #Brogi&Line def 
            ccGrid.append(cc_norm)
        rvGrid = []
        for k in range(len(cc)):
            rv = ((wavelength[0]/paddedWaveTemp[k]) -1 )*2.998*10**5
            rvGrid.append(rv)
    else:
        ccGrid = np.correlate(paddedFluxTemp, flux, mode = 'valid')
        rvGrid = []
        for k in range(len(ccGrid)):
            rv = ((wavelength[0]/paddedWaveTemp[k]) -1 )*2.998*10**5
            rvGrid.append(rv)

    return rvGrid, ccGrid

########################################
# From Dilovan's code
# Performs PCA on a given set of spectra 
# Input: x = flux array, n = number of pca iterations
# Output: x_new = flux array cleaned by pca iterations, comp and comp_ind are the arrays of what was thrown away
########################################
def pca(x,n):

    # Singular value decomposition
    u,s,vt = np.linalg.svd(x,full_matrices=False)
    v = vt.T

	# Isolate first n singular values
    s_new = np.copy(s)
    s_new[0:n]=0.
    s_comp = s-s_new

	# Calculate residuals and (combined) components
    x_new = np.dot(u,np.dot(np.diag(s_new),v.T))+1.0
    comp = np.dot(u,np.dot(np.diag(s_comp),v.T))

	# Also calculate and return each removed component separately
    comp_ind = np.zeros((n,comp.shape[0],comp.shape[1]))	
    for i in range(n):
        s_temp = np.copy(s)
        s_temp[:] = 0.
        s_temp[i] = s[i]
        comp_ind[i,:,:] = np.dot(u,np.dot(np.diag(s_temp),v.T))

    return x_new,comp,comp_ind
    
    
############        
#x values, the mean (mu) and the std (sig) of the gaussian
############
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    
################
#t = observation time in BJD, To = ephemeris in BJD, P = period in days
################
def calculateOrbitalPhase( t, To, P ):
    
    phase = (t - To) / P
    if phase > 1 or phase < -1:
        phase = np.modf(phase)[0]
    if phase > 0.8:
        phase = phase - 1
    if phase < -0.8:
        phase = phase + 1

    return phase

####################
#calculate the radial velocity of the planet at a certian phase (phi)
#input Kp = planet semi amplitude (km/s), phase (phi) ***this assumes you are already in the stars rest frame (have corrected for the barycentric motion and the star's radial velocity)***
#output rv (km/s) 
####################
def calculatePlanetRV( Kp, phi ):

    rv = Kp * np.sin ( 2*np.pi * phi)

    return rv

##################
#injects an absorption signature into the data with a desired RV shift 
##################
def injectPlanetSignalwModel(wave, flux, injectWave, injectRad, StellarRad, rv, transitShape):

    #weight the injected signal by the transit shape
    radInTime = np.sqrt(1 - transitShape) * StellarRad * 6.9634*10**8 / (6.9911*10**7)
    weightFactor = radInTime / np.max(injectRad)
    injectRad_scaled = weightFactor * injectRad
    injectTrans = 1 - ((injectRad_scaled * 6.9911*10**7)**2 / (StellarRad * 6.9634*10**8)**2 )

    #interpolate opacity with an rv shift onto the same grid as the data
    #inject_wave_R , inject_flux_R = convolveToR(injectWave, injectTrans, 80000)
    #inject_wave_R = inject_wave_R[2:-2]
    #inject_flux_R = inject_flux_R[2:-2]
    injectWave_shifted = ((rv / (2.998*10**5)) * injectWave) + injectWave
    #interpolate the shifted template onto the data wavelength grid
    spl = interp1d(injectWave_shifted, injectTrans)
    injectFlux_shifted = spl(wave)
    
    #flux_wSignal = flux + injectFlux_shifted*strength*np.mean(flux)
    flux_wSignal = flux * injectFlux_shifted 

    return flux_wSignal
    
    
##############
#injects an absorption signature into the data with a desired RV shift 
##############
def injectPlanetSignal(wave, flux, injectWave, injectTrans, rv):

    #interpolate opacity with an rv shift onto the same grid as the data
    #inject_wave_R , inject_flux_R = convolveToR(injectWave, injectTrans, 80000)
    #inject_wave_R = inject_wave_R[2:-2]
    #inject_flux_R = inject_flux_R[2:-2]
    injectWave_shifted = ((rv / (2.998*10**5)) * injectWave) + injectWave
    #interpolate the shifted template onto the data wavelength grid
    spl = interp1d(injectWave_shifted, injectTrans)
    injectFlux_shifted = spl(wave)
    
    #flux_wSignal = flux + injectFlux_shifted*10*np.mean(flux)
    flux_wSignal = flux * injectFlux_shifted 

    return flux_wSignal

###############
#calculates the signal to noise ratio of a 1 dimensional flattened cross correlation function
####################
def calculateSNR(rv, cc, cc_injected=[]):

    #subtract the two
    if len(cc_injected) > 0:
        injected_signal = cc_injected - cc
    else:
        injected_signal = cc
        
    #signal without the peak
    no_peak = np.concatenate((injected_signal[0:270], injected_signal[-270:]))
    #get the peak 
    signal = np.max(injected_signal) - np.mean(no_peak)
    #estimate the residual noise at the edges 
    noise = np.std(no_peak)
    #get SNR 
    SNR = signal/noise 

    return SNR

##############
#shifts a 2 dimensional cross correlation function to a desired radial velocity
####################
def shiftXcorl(rvGrid, ccGrid, rv):
    
    restCCs = []
    rvGrid_shifted = np.arange(-150, 150, 0.5) 
    for k in range(len(ccGrid)):
        spl_cc = interp1d(rvGrid - rv[k], ccGrid[k])
        shiftedCC = spl_cc(rvGrid_shifted)
        restCCs.append(shiftedCC)
    #plot to check
    #fig,ax = plt.subplots()
    #ax.imshow(restCCs, cmap=cm.gray, extent = [np.min(rvGrid_shifted), np.max(rvGrid_shifted), 0, len(ccGrid)])
    #plt.show()
    restCCs = np.asarray(restCCs)

    return rvGrid_shifted, restCCs

###############
#
def flattenXcorl(cc):
    
    combined_ccs = np.sum(cc, axis=0)
    
    return combined_ccs


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

#################
#Convolve to a given resolution 
#################
def convolveToR(wavelengths, fluxes, R):

    fluxes = np.asarray(fluxes) 
    deltaWave = np.mean(wavelengths)/ R
    newWavelengths = np.arange(wavelengths[0], wavelengths[-1], deltaWave)
    fwhm = deltaWave / np.mean(wavelengths[1:] - wavelengths[0:-1])
    std = fwhm / ( 2.0 * np.sqrt( 2.0 * np.log(2.0) ) ) #convert FWHM to a standard deviation
    g = Gaussian1DKernel(stddev=std)
    #2. convolve the flux with that gaussian
    convData = convolve(fluxes, g)
    
    #interpolate onto a grid with the requested spacing
    spl = interp1d(wavelengths, convData)
    newFluxes = spl(newWavelengths)

    return newWavelengths, newFluxes

#################
#rotationally broaden the planet's spectrum
#################
def planetVsini(wave, flux, vsini):

    #make an array of parts of a circle evenly spaced in cos theta 
    theta = np.arange(0, np.pi, 0.01)
    cos_theta = np.cos(theta)
    #for a star you need to do this for each radius ring for an exoplanet this is always 1
    #rads = np.arange(0, 1, 0.1)
    rads = [1]

    broad_flux = np.zeros(len(wave))

    for i in range(len(cos_theta)):

        for j in range(len(rads)): 
            #the velocity of this small piece of the surface
            v = vsini * rads[j]* cos_theta[i]
            #doppler shift the lines
            shift_wave = wave / ((v/(2.998*10**5)) + 1)
            #interpolate back onto the original wave grid
            spl_flux = interp1d(shift_wave, flux, fill_value='extrapolate')
            interp_flux = spl_flux(wave)
            #plt.plot(wave, interp_flux)
            broad_flux = broad_flux + interp_flux

    broad_flux = broad_flux/(len(rads)*len(cos_theta))
    #plt.plot(wave, flux)
    #plt.plot(wave, broad_flux)
    #plt.show()

    
    return broad_flux
    
####################
#make the model more realistic by applying rotational broadening, convolving with the instrument profile, and removing the continuum
#wave_range should be in angstroms
####################
def prepareModel(model_atm_wave, model_atm_trans, R, vsini, wave_range):

    #first cut the model so it is only within the wave_range
    model_wave_clip = model_atm_wave[np.where(np.logical_and(model_atm_wave > (wave_range[0]), model_atm_wave < wave_range[-1]))[0]]
    model_trans_clip = model_atm_trans[np.where(np.logical_and(model_atm_wave > (wave_range[0]), model_atm_wave < wave_range[-1]))[0]]

    #Use my original vsini model (method 2), no need to use the new one (method 3)... it takes longer and gives very similar results.  
    #a simple (but incorrect) rotational broadening kernel with a limb darkening coefficient of 0 
    medSpacing = np.median(np.diff(model_wave_clip))
    newModelWave = np.arange(model_wave_clip[0], model_wave_clip[-1], medSpacing)
    modelSpl = interp1d(model_wave_clip, model_trans_clip)
    newModelFlux = modelSpl(newModelWave)

    model_trans_broad = planetVsini(model_wave_clip, model_trans_clip, vsini)

    model_wave_R , model_flux_R = convolveToR(model_wave_clip, model_trans_broad, R)
    model_wave_R = model_wave_R[2:-2]
    model_flux_R = model_flux_R[2:-2]

    return model_wave_R, model_flux_R



####################    
#Pass in:
#an array of wavelength, fluxes, and uncertainty array if one exists
#the model atm wave, the model atm transmission spectrum
#the number of sysrem/pca iterations
#the table that contains all the metadata
#option inputs: plots if you want the plots to be made, you can choose to use pca or sysrem to remove the tellurics (default is pca), and you can choose to return what pca/sysrem threw out instead of what it kept by setting removed to true

#get out:
#radial velocity grid and cross correlation grid between the model and observations after full reduction
####################
def doTheReduction(wave_array, flux_array, model_atm_wave, model_atm_trans, pca_its, table, transitIndices, error_array=None, plots=False):

    if isinstance(error_array, type(None)):
        error_array = 0.1*flux_array

    #interpolate onto an even wavelength grid in velocity space
    #I chose .2 km/s since any larger and the interpolation return inaccurate results
    log_waves = np.arange(np.log10(wave_array[0][10]), np.log10(wave_array[0][-10]), 0.2*np.log10(np.e)/(2.998*10**5))
    wave_grid = 10**log_waves
    interp_flux_grid = []
    interp_sigma_grid = []
    for j in range(len(flux_array)):
        spl_flux = interp1d(wave_array[j], flux_array[j])
        interp_flux = spl_flux(wave_grid)
        spl_sigma = interp1d(wave_array[j], 1/error_array[j])
        interp_sigma = spl_sigma(wave_grid)
        #check if there are any nans
        nans, x = nan_helper(interp_flux)
        interp_flux[nans] = np.interp(x(nans), x(~nans), interp_flux[~nans])
        nans_s, x_s = nan_helper(interp_sigma)
        interp_sigma[nans_s] = np.interp(x_s(nans_s), x_s(~nans_s), interp_sigma[~nans_s])
        #append
        interp_flux_grid.append(interp_flux)
        interp_sigma_grid.append(interp_sigma)

    interp_flux_grid = np.asarray(interp_flux_grid)
    interp_sigma_grid = np.asarray(interp_sigma_grid)

    #STEP 1:normalize the data
    #######
    norm_fluxes = normalize(interp_flux_grid)
    norm_sigmas = normalizeSig(interp_flux_grid, interp_sigma_grid)
    
    #STEP 2: do a 3 sigma clipping along each spectrum (vertical array direction)
    vert_sigma_clip1 = removeVertOutliers(norm_fluxes)
    #vert_sigma_clip1[:, 507076:516355] = 1.0 #sky emission line

    #STEP 4: removed the blaze function by fitting a polynomial
    blaze_removed = []
    for i in range(len(vert_sigma_clip1)):
        poly = np.polyfit(wave_grid, vert_sigma_clip1[i], 3)
        test4 = np.poly1d(poly)
        blaze_removed.append(vert_sigma_clip1[i]/test4(wave_grid))
    blaze_removed = np.asarray(blaze_removed) 

    #see how the clipping removal went
    colors = cm.RdPu_r(np.linspace(0,1,len(flux_array)))
    if plots == True: 
        for j in range(len(flux_array)):
            plt.plot(blaze_removed[j], color = colors[j])
        plt.show()
    

    #STEP 5: get rid of the stellar signal (the stationary signal)
    star_removed = removeHostStar(blaze_removed) 
    #star_removed[:, 507076:516355] = 1.0 #sky emission line

    #do another sigma clipping 
    vert_sigma_clip = removeVertOutliers(star_removed)
    
    #plot to check
    colors = cm.RdPu_r(np.linspace(0,1,len(flux_array)))
    if plots == True: 
        for j in range(len(flux_array)):
            plt.plot(vert_sigma_clip[j], color = colors[j])
        plt.title('star_removed + sigma clipped') 
        plt.show()

    #STEP 6: downweight by the standard deviation
    downweighted = divideByStd(vert_sigma_clip)
    #downweighted[:, 507076:518200] = 1.0 #sky emission line again

    #make sure there are no nans (if nans you will get error in PCA step)
    for k in range(len(downweighted)):
        nans, x = nan_helper(downweighted[k])
        downweighted[k][nans] = np.interp(x(nans), x(~nans), downweighted[k][~nans])

    #STEP 7: remove PCA iterations
    pca_removed, trash1, trash2 = pca(downweighted, pca_its)

    #STEP 8: one more sigma clipping
    vert_sigma_clip2 = removeVertOutliers(pca_removed)

    #remove any shape that appeared in the areas with no data
    #vert_sigma_clip2[:, 507076:518200] = 1.0
    #vert_sigma_clip2[:, 582320:582360] = 1.0 #atm emission line

    if plots == True: 
        colors = cm.RdPu_r(np.linspace(0,1,len(flux_array)))
        for j in range(len(flux_array)):
            plt.plot(vert_sigma_clip2[j], color = colors[j], alpha=alpha)
            plt.title('final-PCA removed') 
        plt.show()


    #plot the reduction steps
    if plots == True:

        label_1 = str(np.round(table['phases'][np.int(np.round(0.25*len(table)))], decimals = 3))
        label_2 = str(np.round(table['phases'][np.int(np.round(0.5*len(table)))], decimals = 3))
        label_3 = str(np.round(table['phases'][np.int(np.round(0.79*len(table)))], decimals = 3))
        y_label_list = [label_3, label_2, label_1]
        
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, sharex=True)
        ax0.plot(wave_grid, norm_fluxes[0], 'k', linewidth=0.5)
        im = ax1.imshow(norm_fluxes, cmap=cm.gray, extent=[np.min(wave_grid), np.max(wave_grid), 0, 1])
        ax2.imshow(blaze_removed, cmap=cm.gray, extent=[np.min(wave_grid), np.max(wave_grid), 0, 1])
        ax3.imshow(vert_sigma_clip2, cmap=cm.gray, extent=[np.min(wave_grid), np.max(wave_grid), 0, 1])
        ax3.set_xlabel('Wavelength ($\AA$)')
        ax2.set_ylabel('Phase')
        ax0.set_ylabel('Norm Flux')
        ax1.set_yticks([0.25,0.5,0.75])
        ax2.set_yticks([0.25,0.5,0.75])
        ax3.set_yticks([0.25,0.5,0.75])
        ax1.set_yticklabels(y_label_list)
        ax2.set_yticklabels(y_label_list)
        ax3.set_yticklabels(y_label_list)
        ax0.set_aspect(190)
        ax1.set_aspect(350)
        ax2.set_aspect(350)
        ax3.set_aspect(350)
        plt.show()

    #cross correlate the model atm with the data
    rv_grid, cc_grid = crossCorrelate(wave_grid, vert_sigma_clip2, model_atm_wave, model_atm_trans)
    rv_grid = np.asarray(rv_grid)
    cc_grid = np.asarray(cc_grid)

    #rv grid goes in wrong direction, flip it
    if rv_grid[0] > rv_grid[1]:
        rv_grid = rv_grid[::-1]
        for c in range(len(cc_grid)):
            cc_grid[c] = cc_grid[c][::-1]

    #return the normalized cc grid and the rv grid
    return rv_grid, cc_grid

#################
#make the 2d Kp versus Vsys plot like in lots of papers
#find the best Vsys and Kp 
#################
def findKpVsys(rv_grid, cc_grid, Kp_orig, phases): 

    Kps = np.arange(Kp_orig + 80,-100, -2)
    Vsys = np.arange(-100, 100, 1)
    rvGrid_shifted = np.arange(-100, 100, 1)
    SNRs = []
    for Kp in Kps:
         snrs = []
         for V in Vsys:
             #radVel = calculatePlanetRV(Kp, t['phases']) + V
             shiftedCCs = []
             for c in range(len(cc_grid)):
                 spl_cc = interp1d(rv_grid - (calculatePlanetRV(Kp, phases[c])+V), cc_grid[c])
                 shiftedCC = spl_cc(rvGrid_shifted)
                 shiftedCCs.append(shiftedCC)
             combinedCCs = np.sum(shiftedCCs, axis=0)
             CCatZero = combinedCCs[np.where(rvGrid_shifted == 0)] 
             stdCC = np.std(np.concatenate((combinedCCs[0:70], combinedCCs[-70:])))
             SNR = (CCatZero-np.mean(combinedCCs)) / stdCC
             snrs.append(SNR[0])
         SNRs.append(snrs) 

    SNRs_a = np.asarray(SNRs)
    #np.save('KpVsVsys_WASP33_VIS.npy',SNRs_a)
    fig, ax = plt.subplots(figsize=(8,5))
    im = ax.imshow(SNRs_a, cmap = cm.OrRd, extent=[Vsys[0], Vsys[-1], Kps[-1], Kps[0]])
    ax.set_xlabel('$V_{wind}$ (km s$^{-1}$)', fontsize=14)
    ax.set_ylabel('$K_p$ (km s$^{-1}$)', fontsize=14)
    ax.set_xlim(-100,100)
    ax.set_ylim(-0, 250)
    ax.set_aspect(0.6)
    cb = fig.colorbar(im)
    cb.set_label('SNR', rotation=270, fontsize=14)
    cb.ax.tick_params(labelsize=14)
    cb.ax.get_yaxis().labelpad = 15
    plt.plot([-100,100], [Kp_orig, Kp_orig], 'k--')
    plt.plot([0,0], [Kps[0], Kps[-1]], 'k--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Ca', fontsize=14)
    #plt.savefig('Masc2_KpVsVsys_8PCA_-7VMRH-.pdf') 
    plt.show()
    pdb.set_trace()


    index = np.unravel_index(np.argmax(SNRs_a, axis=None), SNRs_a.shape)
    best_kp = Kps[index[0]]
    best_vsys = Vsys[index[1]]
    #np.save('SNR_map_20180712_2500-8VMR.npy', SNRs_a) 

    return best_kp, best_vsys

#find the best Vsys and Kp 
def findKpVsyswModel(rv_grid, cc_grid, Kp_orig, phases, transitShape): 

    Kps = np.arange(Kp_orig + 80,-100, -2)
    Vsys = np.arange(-100, 100, 1)
    rvGrid_shifted = np.arange(-100, 100, 1)
    SNRs = []
    for Kp in Kps:
         snrs = []
         for V in Vsys:
             #radVel = calculatePlanetRV(Kp, t['phases']) + V
             shiftedCCs = []
             for c in range(len(cc_grid)):
                 spl_cc = interp1d(rv_grid - (calculatePlanetRV(Kp, phases[c])+V), cc_grid[c])
                 shiftedCC = spl_cc(rvGrid_shifted)
                 shiftedCCs.append(shiftedCC)
             combinedCCs = flattenXcorlwModel(shiftedCCs, transitShape)
             CCatZero = combinedCCs[np.where(rvGrid_shifted == 0)] 
             stdCC = np.std(np.concatenate((combinedCCs[0:70], combinedCCs[-70:])))
             SNR = (CCatZero-np.mean(combinedCCs)) / stdCC
             snrs.append(SNR[0])
         SNRs.append(snrs) 

    SNRs_a = np.asarray(SNRs)
    #np.save('KpVsVsys_WASP33_VIS_wModel.npy',SNRs_a)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(SNRs_a, cmap = cm.OrRd, extent=[Vsys[0], Vsys[-1], Kps[-1], Kps[0]])
    ax.set_xlabel('$V_{wind}$ (km s$^{-1}$)', fontsize=14)
    ax.set_ylabel('$K_p$ (km s$^{-1}$)', fontsize=14)
    ax.set_xlim(-100,100)
    ax.set_aspect(0.65)
    cb = fig.colorbar(im)
    cb.set_label('SNR', rotation=270, fontsize=14)
    cb.ax.tick_params(labelsize=14)
    cb.ax.get_yaxis().labelpad = 15
    plt.plot([-100,100], [Kp_orig, Kp_orig], 'k--')
    plt.plot([0,0], [Kps[0], Kps[-1]], 'k--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('Masc2_KpVsVsys_wModel_-7VMR_4PCA_wMask.pdf') 
    plt.show()
    pdb.set_trace()


    index = np.unravel_index(np.argmax(SNRs_a, axis=None), SNRs_a.shape)
    best_kp = Kps[index[0]]
    best_vsys = Vsys[index[1]]
    #np.save('SNR_map_20180712_2500-8VMR.npy', SNRs_a) 

    return best_kp, best_vsys
