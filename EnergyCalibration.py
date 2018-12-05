from Spectrum import Spectrum
import numpy as np
import matplotlib.pyplot as plt

# Creating Cobalt-60 object
co60 = Spectrum()
co60.read_from_csv("18-10-18_Co60.csv")

# Creating Bismuth-207 object
bi207 = Spectrum()
bi207.read_from_csv("25-10-18_Bi207.csv")

channel_peaks = np.array([])
d_channel_peaks = np.array([])

# Finding Gaussian parameters for each peak of co60.
co60_peaks = co60.find_peaks(show=False)
for i in co60_peaks:
    popt, pcov = co60.fit_3sigma_gaussian(i, "AllParams", show=False)
    channel_peaks = np.append(channel_peaks, popt[1])
    d_channel_peaks = np.append(d_channel_peaks, np.diag(pcov)[1])

# Finding peaks for Bi-207
bi207_peaks = np.array([])
bi207_peaks = bi207.find_peaks(threshold=0.3, show=False)
# First two are X-ray
bi207_peaks = bi207_peaks[2:]
# Manually adding the last one
bi207_peaks = np.append(bi207_peaks, 2322)
for i in bi207_peaks:
    popt, pcov = bi207.fit_3sigma_gaussian(i, "AllParams", show=False)
    print(popt)
    channel_peaks = np.append(channel_peaks, popt[1])
    d_channel_peaks = np.append(d_channel_peaks, np.diag(pcov)[1])

# Manually writing in known energy peaks.
# 2 x Co-60, 3 x Bi-207
their_energies = np.array([1173.228, 1332.492,
                           569.698, 1063.656, 1770.228])

Spectrum.polynomial_calibration(channel_peaks, their_energies)
print(Spectrum.calibration_a)
print(Spectrum.calibration_b)
print(Spectrum.calibration_c)

co60.apply_polynomial_calibration()
#co60.plot(calibrated=True)

bi207.apply_polynomial_calibration()
#bi207.plot(calibrated=True)


