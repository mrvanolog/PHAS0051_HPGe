from Spectrum import Spectrum
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Initialising calibration factors
Spectrum.calibration_a = -5.677763487790322e-08
Spectrum.calibration_b = 0.7626393415958842
Spectrum.calibration_c = -0.7553693869647748

# Creating Cobalt-60 object
co60 = Spectrum()
co60.read_from_csv("18-10-18_Co60.csv")
#co60.read_from_csv("191118-Co60-SHIELD-HIGHVOLT-FULL.csv")
co60.apply_polynomial_calibration()

# Creating Bismuth-207 object
bi207 = Spectrum()
bi207.read_from_csv("25-10-18_Bi207.csv")
bi207.apply_polynomial_calibration()

### Finding areas of peaks ###
areas = np.array([])
# Cobalt:
peaks = co60.find_peaks(show=False)
for i in peaks:
    area = co60.fit_3sigma_gaussian(i, mode="Area", show=False)
    areas = np.append(areas, area)

# Bismuth:
bi207_peaks = bi207.find_peaks(threshold=0.3, show=False)
bi207_peaks = bi207_peaks[2:]
for i in bi207_peaks:
    area = bi207.fit_3sigma_gaussian(i, mode="Area", show=False)
    areas = np.append(areas, area)

area = bi207.fit_3sigma_gaussian(2322, mode="Area", show=False)
areas = np.append(areas, area)
print ("Areas", areas)

### Calculating efficiency ###
# Array of real times for each peak
t = np.array([co60.real_time, co60.real_time, bi207.real_time, bi207.real_time, bi207.real_time])
print("Times:", t)

def abs_efficiency(Y,A,t,Area):
    """Calculates efficiency"""
    N_emitted = A * Y * t  # calculates number of emitted gamma rays
    print("Emitted", N_emitted)
    N_measured = Area      # number of measured gamma rays
    print("Measured", N_measured)
    eff_abs = N_measured / N_emitted  # calculates efficiency
    return eff_abs


def int_efficiency(Y, A, t, area, d, r):
    """Calculates intrinsic efficiency"""
    solid_angle = 2*np.pi*(1 - (d/np.sqrt(d**2+r**2)))
    n_through = A*Y*t * solid_angle/(4*np.pi)
    eff_int = area/n_through
    return eff_int


Y_co = 2      # gamma ray yield for cobalt-60
A_co = 1037   # activity for cobalt-60
Y_bi = 3      # gamma ray yield for bismuth-207
A_bi = 32000  # activity for bismuth-207

Y = np.array([0.9985, 0.999826, 0.9775, 0.745, 0.0687])
Y = Y*100
A = np.array([A_co, A_co, A_bi, A_bi, A_bi])

abs_eff = abs_efficiency(Y, A, t, areas)
print("Absolute Efficiency:", abs_eff)

d = 5e-3 + 5.23e-3  # distance from the documentation + thickness of the box
eff_area = 29.1e-4
# Assuming area a circular, A=Pi*r^2
r = np.sqrt(eff_area/np.pi)

int_eff = int_efficiency(Y, A, t, areas, d, r)
print("Intrinsic Efficiency:", int_eff)

# Manually writing in known energy peaks.
# 2 x Co-60, 3 x Bi-207
their_energies = np.array([1173.228, 1332.492,
                           569.698, 1063.656, 1770.228])


# Defining a curve to fit
def polynomial3(x, b, c, d):
    return b*x**2 + c*x + d


guess = [1, 1, 1]
x_curve = np.linspace(500, 1800, 1300)
popt, pcov = sp.optimize.curve_fit(polynomial3, their_energies[2:], abs_eff[2:], guess)
y_curve = polynomial3(x_curve, popt[0], popt[1], popt[2])

plt.figure()
plt.title("Photopeak efficiency dependence on energy.")
plt.xlabel("Energy, Kev")
plt.ylabel("Efficiency")
plt.plot(their_energies, abs_eff, "k.")
#plt.plot(x_curve, y_curve, "r--", linewidth=0.5)
plt.grid(True, linestyle='--', linewidth=0.2)
plt.show()

plt.figure()
plt.title("log-log Photopeak efficiency dependence on energy.")
plt.xlabel("Energy, Kev")
plt.ylabel("Efficiency")
plt.loglog(their_energies[2:], abs_eff[2:], "k.")
#plt.plot(x_curve, y_curve, "r--", linewidth=0.5)
plt.grid(True, linestyle='--', linewidth=0.2)
plt.show()














'''
# USING DATA FROM SOME OTHER GUYS
some_efficiency1 = np.array([abs_eff[0], 0.0043, 0.0013, 0.0005])
some_efficiency2 = np.array([abs_eff[1], 0.0038, 0.0012, 0.0005])
distance1 = np.array([d, 5, 10, 15])

plt.figure()
plt.plot(distance1, some_efficiency1, "r.")
plt.plot(distance1, some_efficiency2, "g.")
plt.xlim(0)
#plt.ylim(0, 0.05)
plt.show()
'''
