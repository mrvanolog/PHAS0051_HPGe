from Spectrum import Spectrum
import numpy as np
import matplotlib.pyplot as plt
import peakutils

### BACKGROUND REDUCTION TEST ###

Spectrum.calibration_a = -5.677763487790322e-08
Spectrum.calibration_b = 0.7626393415958842
Spectrum.calibration_c = -0.7553693869647748

# Creating Cobalt-60 object
co60 = Spectrum()
co60.read_from_csv("18-10-18_Co60.csv")
co60.apply_polynomial_calibration()

# Creating Bismuth-207 object
bi207 = Spectrum()
bi207.read_from_csv("25-10-18_Bi207.csv")
bi207.apply_polynomial_calibration()

# Creating background object, with the cap
bkgd_capon = Spectrum()
bkgd_capon.read_from_csv("25-10-18_BKGD_after_Bi207_CapOn.csv")
bkgd_capon.apply_polynomial_calibration()

# Creating background object, without the cap
bkgd_capoff = Spectrum()
bkgd_capoff.read_from_csv("25-10-18_BKGD_after_Bi207_CapOff.csv")
bkgd_capoff.apply_polynomial_calibration()


# Reducing background background
test = Spectrum.reduce_background(bkgd_capon, bkgd_capoff)
test.plot(calibrated=False, log=False)

plt.figure()
plt.plot(test.counts, "k.")
plt.show()

### TRYING TO ELIMINATE BASELINE ###
'''

x = bi207.channels[747-200:747+200]  # 747, 1395
y = bi207.counts[747-200:747+200]

y_base = peakutils.baseline(y, deg=4, max_it=100, tol=1e-3)

#y_minusbase = np.append(y[:1700]-y_base[:1700], y[1700:])

plt.figure(figsize=(20,15))
plt.plot(x, y, 'k--.')
plt.plot(x, y_base, 'r-')
#plt.plot(x, y - y_base, 'g-')
plt.ylim(0, 7e3)
plt.show()
'''
