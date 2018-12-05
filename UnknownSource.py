from Spectrum import Spectrum
import numpy as np

us = Spectrum()
us.read_from_csv("IAEA_sample_present_20.11.18.csv")

bkgd_us = Spectrum()
bkgd_us.read_from_csv("IAEA_sample_not_present_22.11.18.csv")

# Calibration factors from least squares fit
Spectrum.calibration_a = -5.677763487790322e-08
Spectrum.calibration_b = 0.7626393415958842
Spectrum.calibration_c = -0.7553693869647748
us.apply_polynomial_calibration()

us.plot(calibrated=False, log=False) # plots unknown source spectra
us_reduced = Spectrum.reduce_background(us, bkgd_us) # reduces background from spectrum

means = np.array([])
vars = np.array([])
dmeans = np.array([])
dvars = np.array([])


peaks = us_reduced.find_peaks(abs_threshold=True, threshold=500)
peaks = peaks[-5:]
print(peaks)



# gaussians plotted for 5 peaks highest in energy
for i in peaks:
    popt, pcov = us.fit_3sigma_gaussian(i, "AllParams", show=False)
    means = np.append(means, popt[1])
    dmeans = np.append(dmeans, np.diag(pcov)[1])
    vars = np.append(vars, popt[2])
    dvars = np.append(dvars, np.diag(pcov)[2])

means_kev = Spectrum.channel_to_energy(means)
print(means_kev)
#
# Tl-208, Bi-214, Cs-137 / Ba-137m, Ac-228, K-40
#

def manual_peaks(ch_guess):
    """Allows peaks to be found manually"""
    popt1, pcov1 = us.fit_3sigma_gaussian(ch_guess, "AllParams",show=True)
    mean1 = popt1[1]
    dmeans1 = np.diag(pcov1)[1]
    vars1 = popt1[2]
    dvars1 = np.diag(pcov1)[2]
    return mean1, dmeans1, vars1, dvars1


# Peak around 461 channel, 351 energy
# Bi-211
A = manual_peaks(461)
#print(A[0]) # mean
print(Spectrum.channel_to_energy(A[0]))

#  Peak around channel 313
# Pb-212
B = manual_peaks(313)
#print(B[0]) # mean
print(Spectrum.channel_to_energy(B[0]))

# Peaks around channel 388
# Pb-214
C = manual_peaks(388)
##print(C[0]) # mean
print(Spectrum.channel_to_energy(C[0]))


# peaks around channel 3428
# Tl-208 (Thalium)
D = manual_peaks(3428)
#print(D[0])
print(Spectrum.channel_to_energy(D[0]))


# peaks around channel 2315
# Bi-214
E = manual_peaks(2315)
#print(E[0])
print(Spectrum.channel_to_energy(E[0]))

