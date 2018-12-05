from Spectrum import Spectrum
import numpy as np
import matplotlib.pyplot as plt

# Initialising calibration factors
Spectrum.calibration_a = -5.677763487790322e-08
Spectrum.calibration_b = 0.7626393415958842
Spectrum.calibration_c = -0.7553693869647748

# Creating Cobalt-60 object
co60 = Spectrum()
co60.read_from_csv("18-10-18_Co60.csv")
co60.apply_polynomial_calibration()
co60.plot()

# Creating Bismuth-207 object
bi207 = Spectrum()
bi207.read_from_csv("25-10-18_Bi207.csv")
bi207.apply_polynomial_calibration()

# Creating an unknown source object
us = Spectrum()
us.read_from_csv("IAEA_sample_present_20.11.18.csv")
us.apply_polynomial_calibration()
# And its background
bkgd_us = Spectrum()
bkgd_us.read_from_csv("IAEA_sample_not_present_22.11.18.csv")
# Reducing the background.
#us.plot(calibrated=False, log=False) # plots unknown source spectra
us_reduced = Spectrum.reduce_background(us, bkgd_us)

### Finding FWHM OF PEAKS ###
# Initialising arrays
our_energies = np.array([])
d_our_energies = np.array([])
FWHM_array = np.array([])
d_FWHM_array = np.array([])

# COBALT #
peaks = co60.find_peaks(show=False)
for i in peaks:
    FWHM, d_FWHM = co60.fit_3sigma_gaussian(i, mode="FWHM", show=False)
    mean, d_mean = co60.fit_3sigma_gaussian(i, mode="Mean", show=False)
    FWHM_array = np.append(FWHM_array, FWHM)
    d_FWHM_array = np.append(d_FWHM_array, d_FWHM)
    our_energies = np.append(our_energies, mean)
    d_our_energies = np.append(d_our_energies, d_mean)

# BISMUTH #
bi207_peaks = bi207.find_peaks(threshold=0.3, show=False)
#bi207.plot(calibrated=False, xMin=bi207_peaks[0]-10, xMax=bi207_peaks[0]+10)
# Second X-Ray peak
FWHM, d_FWHM = bi207.fit_gaussian(bi207.channels[bi207_peaks[1]-2:bi207_peaks[1]+3],
                                  bi207.counts[bi207_peaks[1]-2:bi207_peaks[1]+3],
                                  mode="FWHM", show=False)
mean, d_mean = bi207.fit_gaussian(bi207.channels[bi207_peaks[1]-2:bi207_peaks[1]+3],
                                  bi207.counts[bi207_peaks[1]-2:bi207_peaks[1]+3],
                                  mode="Mean", show=False)
print("X-Ray 2:", FWHM, "+-", d_FWHM)

FWHM_array = np.append(FWHM_array, FWHM)
d_FWHM_array = np.append(d_FWHM_array, d_FWHM)
our_energies = np.append(our_energies, mean)
d_our_energies = np.append(d_our_energies, d_mean)

# this is for uncertainties of first X-Ray
d1 = d_mean/mean
d2 = d_FWHM/FWHM

# First X-Ray peak
FWHM, d_FWHM = bi207.fit_gaussian(bi207.channels[bi207_peaks[0]-3:bi207_peaks[0]+2],
                                  bi207.counts[bi207_peaks[0]-3:bi207_peaks[0]+2],
                                  mode="FWHM", show=False)
mean, d_mean = bi207.fit_gaussian(bi207.channels[bi207_peaks[0]-3:bi207_peaks[0]+2],
                                  bi207.counts[bi207_peaks[0]-3:bi207_peaks[0]+2],
                                  mode="Mean", show=False)

# Due to only one datapoint to the right of the peak,
# I assume same % uncertainty as the neighbouring peak
FWHM_array = np.append(FWHM_array, FWHM)
d_FWHM_array = np.append(d_FWHM_array, d2*FWHM)
our_energies = np.append(our_energies, mean)
d_our_energies = np.append(d_our_energies, d1*mean)

print("X-Ray 1:", FWHM, "+-", d2*FWHM)

# Other peaks
bi207_peaks = bi207_peaks[2:]
for i in bi207_peaks:
    FWHM, d_FWHM = bi207.fit_3sigma_gaussian(i, mode="FWHM", show=False)
    mean, d_mean = bi207.fit_3sigma_gaussian(i, mode="Mean", show=False)
    FWHM_array = np.append(FWHM_array, FWHM)
    d_FWHM_array = np.append(d_FWHM_array, d_FWHM)
    our_energies = np.append(our_energies, mean)
    d_our_energies = np.append(d_our_energies, d_mean)

# Last peak
FWHM, d_FWHM = bi207.fit_3sigma_gaussian(2322, mode="FWHM", show=False)
mean, d_mean = bi207.fit_3sigma_gaussian(2322, mode="Mean", show=False)
FWHM_array = np.append(FWHM_array, FWHM)
d_FWHM_array = np.append(d_FWHM_array, d_FWHM)
our_energies = np.append(our_energies, mean)
d_our_energies = np.append(d_our_energies, d_mean)

# UNKNOWN SOURCE #
us_peaks = us_reduced.find_peaks(abs_threshold=True, threshold=500)
us_peaks = us_peaks[-5:]

# gaussians plotted for 5 peaks highest in energy
for i in us_peaks:
    FWHM, d_FWHM = us.fit_3sigma_gaussian(i, mode="FWHM", show=False)
    mean, d_mean = us.fit_3sigma_gaussian(i, mode="Mean", show=False)
    FWHM_array = np.append(FWHM_array, FWHM)
    d_FWHM_array = np.append(d_FWHM_array, d_FWHM)
    our_energies = np.append(our_energies, mean)
    d_our_energies = np.append(d_our_energies, d_mean)
#
# Tl-208, Bi-214, Cs-137 / Ba-137m, something, K-40
#

# Peak around 461 channel, 351 energy
# Bi-211
FWHM, d_FWHM = us.fit_3sigma_gaussian(461, mode="FWHM", show=False)
mean, d_mean = us.fit_3sigma_gaussian(461, mode="Mean", show=False)
FWHM_array = np.append(FWHM_array, FWHM)
d_FWHM_array = np.append(d_FWHM_array, d_FWHM)
our_energies = np.append(our_energies, mean)
d_our_energies = np.append(d_our_energies, d_mean)

#  Peak around channel 313
# Pb-212
FWHM, d_FWHM = us.fit_3sigma_gaussian(313, mode="FWHM", show=False)
mean, d_mean = us.fit_3sigma_gaussian(313, mode="Mean", show=False)
FWHM_array = np.append(FWHM_array, FWHM)
d_FWHM_array = np.append(d_FWHM_array, d_FWHM)
our_energies = np.append(our_energies, mean)
d_our_energies = np.append(d_our_energies, d_mean)

# Peaks around channel 388
# Pb-214
FWHM, d_FWHM = us.fit_3sigma_gaussian(388, mode="FWHM", show=False)
mean, d_mean = us.fit_3sigma_gaussian(388, mode="Mean", show=False)
FWHM_array = np.append(FWHM_array, FWHM)
d_FWHM_array = np.append(d_FWHM_array, d_FWHM)
our_energies = np.append(our_energies, mean)
d_our_energies = np.append(d_our_energies, d_mean)

# peaks around channel 3428
# Tl-208 (Thalium)
FWHM, d_FWHM = us.fit_3sigma_gaussian(3428, mode="FWHM", show=False)
mean, d_mean = us.fit_3sigma_gaussian(3428, mode="Mean", show=False)
FWHM_array = np.append(FWHM_array, FWHM)
d_FWHM_array = np.append(d_FWHM_array, d_FWHM)
our_energies = np.append(our_energies, mean)
d_our_energies = np.append(d_our_energies, d_mean)

# peaks around channel 2315
# Bi-214
FWHM, d_FWHM = us.fit_3sigma_gaussian(2315, mode="FWHM", show=False)
mean, d_mean = us.fit_3sigma_gaussian(2315, mode="Mean", show=False)
FWHM_array = np.append(FWHM_array, FWHM)
d_FWHM_array = np.append(d_FWHM_array, d_FWHM)
our_energies = np.append(our_energies, mean)
d_our_energies = np.append(d_our_energies, d_mean)

# Converting channels to energies
Spectrum.channel_to_energy(our_energies)
d_our_energies = np.add(d_our_energies, 0.62)
print("FWHM:", FWHM_array)
print("d_FWHM:", d_FWHM_array)
print("Our Energies:", our_energies)

# Plotting FWHM againts energy
plt.figure(figsize=(8, 8))
plt.title("FWHM dependence on energies of the peaks.")
plt.errorbar(our_energies, FWHM_array, xerr=d_our_energies, yerr=d_FWHM_array, fmt="k.", label="Data Points")
plt.grid(True, linestyle='--', linewidth=0.2)
plt.legend(loc="upper left")
plt.xlim(0, np.amax(our_energies)*1.1)
plt.xlabel("Energy, KeV")
plt.ylabel("FWHM, KeV")
plt.show()

# Calculating resolution
resolution = FWHM/our_energies * 100
d_resolution = resolution*np.sqrt((d_FWHM_array/FWHM)**2 + (d_our_energies/our_energies)**2)

# Plotting it
plt.figure(figsize=(5, 5))
plt.errorbar(our_energies, resolution, xerr=d_our_energies, yerr=d_resolution,
             fmt="k.", linewidth=0.4, markersize=2)
plt.grid(True, linestyle='--', linewidth=0.2)
plt.xlim(0, np.amax(our_energies)*1.1)
plt.ylim(0)
plt.xlabel("Energy, KeV")
plt.ylabel("Resolution")
plt.show()