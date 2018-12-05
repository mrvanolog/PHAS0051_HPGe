from Spectrum import Spectrum
import numpy as np

### ANALYSING PEAKS OF THE BACKGROUND RADIATION ###

# create background object
bkgd = Spectrum()
bkgd.read_from_csv("IAEA_sample_not_present_22.11.18.csv")

Spectrum.calibration_a = -5.677763487790322e-08
Spectrum.calibration_b = 0.7626393415958842
Spectrum.calibration_c = -0.7553693869647748
bkgd.apply_polynomial_calibration()

bkgd.plot(calibrated=False)

# create arrays for positions and uncertainties
# of background peaks in channels
means = np.array([])
dmeans = np.array([])
vars = np.array([])
dvars = np.array([])

# Peak around 313 channel, 238 energy
mean, dmean = bkgd.fit_3sigma_gaussian(313, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# Peak around 317 channel, 242 energy
# !AMPLITUDE IS 2 TIMES LARGER THAN IT SHOULD BE, CONSIDER INCREASING THE UNCERTAINTY! #
# > UNCERTAINTY IS HALF A BIN = 0.5 "
bkgd.plot(xMin=235, xMax=249, yMax=1.5e4)
mean, dmean = bkgd.fit_gaussian(bkgd.channels[317-2:317+3], bkgd.counts[317-2:317+3], "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, 0.5)

# Peak around 387 channel, 295 energy
mean, dmean = bkgd.fit_3sigma_gaussian(387, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# Peak around 444 channel, 338 energy
mean, dmean = bkgd.fit_3sigma_gaussian(444, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# Peak around 461 channel, 351 energy
mean, dmean = bkgd.fit_3sigma_gaussian(461, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)


# Peak around 671 channel, 511 energy
popt, pcov = bkgd.fit_3sigma_gaussian(671, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# Peak around 765 channel, 583 energy
mean, dmean = bkgd.fit_3sigma_gaussian(765, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# Peak around 800 channel, 609 energy
mean, dmean = bkgd.fit_3sigma_gaussian(800, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# Peak around 1196 channel, 911 energy
mean, dmean = bkgd.fit_3sigma_gaussian(1196, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# Peak around 1272 channel, 969 energy
mean, dmean = bkgd.fit_3sigma_gaussian(1272, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# Peak around 1917 channel, 1461 energy
mean, dmean = bkgd.fit_3sigma_gaussian(1917, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# Peak around 2316 channel, 1765 energy
mean, dmean = bkgd.fit_3sigma_gaussian(2316, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# Peak around 3429 channel, 2614 energy
mean, dmean = bkgd.fit_3sigma_gaussian(3429, "Mean", show=False)
means = np.append(means, mean)
dmeans = np.append(dmeans, dmean)

# UNCERTAINTIES #
print("Channels means:", means)
print("Uncertainties:", dmeans)

# convert from channels to energy
energies = Spectrum.channel_to_energy(means)
upper_bound = Spectrum.channel_to_energy(np.add(means, dmeans))
lower_bound = Spectrum.channel_to_energy(np.add(means, -dmeans))
d_energies = np.multiply(np.add(upper_bound, -lower_bound), 0.5)
d_energies = np.add(d_energies, np.mean(bkgd.d_energies))

print("Energies means:", energies)
print("Uncertainty:", d_energies)
