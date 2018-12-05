# PREREQUISITES:
# peakutils

import numpy as np
import pandas as pd
import peakutils
from scipy.interpolate import UnivariateSpline
import scipy as sp
import matplotlib as mpl
mpl.use('module://backend_interagg')
import matplotlib.pyplot as plt


class Spectrum(object):
    """Class describing the radioactive spectrum as detected on HPGe"""
    calibration_a = 0
    calibration_b = 1
    calibration_c = 0

    d_a = 0
    d_b = 0
    d_c = 0

    def __init__(self):
        self.channels = np.array([])
        self.energies = np.array([])
        self.d_energies = np.array([])
        self.counts = np.array([])
        self.real_time = 0          # Time elapsed since the start of the program, s
        self.live_time = 0          # Operational time of the detector, s
        self.datetime = 0           # WIP

    def read_from_csv(self, filename):
        """
        Extracts the spectrum data from a .csv file.
        Requires the file to follow the format of *WIP* program that we use to read the spectrum.
        :param filename: the sting with the name of the .csv file to be read.
        :return: updates:
                    - real_time
                    - live_time
                    - channels
                    - energies
                    - counts
        """
        data = pd.read_csv(filename, names=["Channel", "Energy", "Counts", "del"])
        self.live_time = float(data.iat[2, 1])
        self.real_time = float(data.iat[3, 1])
        data = data.drop(data.index[[0, 1, 2, 3, 4, 5, 6]])
        del data["del"]
        data = data.apply(pd.to_numeric, errors="ignore")
        data = np.array(data.values)
        self.channels = data[:, 0].astype(int)
        self.energies = data[:, 1]
        self.counts = data[:, 2].astype(int)

    def plot(self, xMin=None, xMax=None, yMin=None, yMax=None, calibrated=True, log=False):
        """
        Plots the spectrum on the screen. Can plot counts against energies or channels.
        Custom limits can be used to get a zoom-in.
        Log of counts can be plotted instead.
        :param xMin: lower boundary of x-value to be plotted.
        :param xMax: upper boundary of x-value to be plotted.
        :param yMin: lower boundary of y-value to be plotted.
        :param yMax: upper boundary of y-value to be plotted.
        :param calibrated: if True, plots against energy. Otherwise against channels.
        :param log: if True, plots log(counts). Otherwise just counts.
        :return: Plot of the spectrum.
        """
        plt.figure(figsize=(12, 8))

        # If the energy is calbirated, show it on x-scale
        # Otherwise plot against channels
        if calibrated:
            xVal = self.energies
        else:
            xVal = self.channels

        # If no custom xMax/xMin is input, set as the maximum/minimum values.
        if xMax is None:
            xMax = np.amax(xVal)
        if xMin is None:
            xMin = np.amin(xVal)

        #  Set the number of ticks on both axes
        plt.locator_params(axis='x', nbins=20)
        plt.locator_params(axis='y', nbins=10)

        # Takes the "Counts" column
        yVal = self.counts
        # Plots log of counts if log=True
        if log:
            plt.yscale('log')
            if yMin is None:
                yMin = 1

        # If no custom yMax/yMin is input, set as the just above maximum count number/zero.
        if yMax is None:
            yMax = np.amax(yVal)*1.1
        if yMin is None:
            yMin = 0

        plt.plot(xVal, yVal, "k--.", markersize=2, linewidth=0.35)
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
        plt.ylabel("Counts")
        if calibrated:
            plt.xlabel("Energy, KeV")
        else:
            plt.xlabel("Channel")
        plt.grid(True, linestyle='--', linewidth=0.2)
        plt.show()

    def find_peaks(self, abs_threshold=False, threshold=0.7, show=True):
        """
        :param abs_threshold: Boolean parameter of whether the threshold value is absolute or normalised.
        :param threshold: Limit below which peaks will be ignored.
        :param show: If True, show determined peaks on the plot.
        :return:
        """

        #  peakutils.indexes finds the numeric index of the peaks in *y*
        #  by taking its first order difference.
        peaks = peakutils.indexes(self.counts, thres_abs=abs_threshold, thres=threshold)
        if show:
            plt.figure(figsize=(12, 8))
            plt.plot(self.counts, 'k.', markersize=1, linewidth=0.4)
            plt.scatter(peaks, self.counts[peaks], marker='x', color='g', s=40)
            plt.grid(True, linestyle='--', linewidth=0.2)
            plt.xlim(0, 4096)
            plt.show()
        return peaks

    @staticmethod
    def fit_gaussian(x_data, y_data, mode, show=True):

        #  Define the gaussian function to input into scipy.optimize.curve_fit
        def gaussian(x, ampl, center, dev, elev):
            """
            Computes the Gaussian function.
            :param x: Point to evaluate the Gaussian for.
            :param ampl: Amplitude.
            :param center: Center.
            :param dev: Width.
            :param elev: Elevation above ground.
            :return: Value of the specified Gaussian at *x*
            """
            return ampl * np.exp(-(x - center) ** 2 / (2 * dev ** 2)) + elev

        guess = [np.amax(y_data), x_data[int(np.floor(np.size(x_data)/2))], 1, 100]

        popt, pcov = sp.optimize.curve_fit(gaussian, x_data, y_data, guess)
        ampl = popt[0]
        dampl = np.sqrt(np.diag(pcov)[0])
        mean = popt[1]
        dmean = np.sqrt(np.diag(pcov)[1])
        stdev = popt[2]
        dstdev = np.sqrt(np.diag(pcov)[2])
        elev = popt[3]
        delev = np.sqrt(np.diag(pcov)[3])

        if show:
            x_gaussian = np.linspace(np.amin(x_data), np.amax(x_data), 200)
            y_gaussian = gaussian(x_gaussian, ampl, mean, stdev, elev)
            plt.figure()
            plt.plot(x_data, y_data, 'k.', markersize=1)
            plt.plot(x_gaussian, y_gaussian, 'r-', linewidth=0.5)
            plt.grid(True, linestyle='--', linewidth=0.2)

            plt.axhline(y=popt[3], color='g', linestyle='-', linewidth=0.35)

            plt.show()

        if mode == "Amplitude":
            return ampl, dampl

        elif mode == "Mean":
            return mean, dmean

        elif mode == "Variance":
            return stdev**2, 2*stdev*dstdev

        elif mode == "AllParams":
            return popt, pcov

        elif mode == "FWHM":
            FWHM = 2*np.sqrt(2*np.log(2))*stdev
            d_FWHM = 2 * np.sqrt(2 * np.log(2)) * dstdev
            return FWHM, d_FWHM

        else:
            print("Choose a valid mode from on of the following options: Mean, Var, FWHM")
            return

    def fit_3sigma_gaussian(self, x_peak, mode, show=True):

        #  Define the gaussian function to input into scipy.optimize.curve_fit
        def gaussian(x, ampl, center, dev, elev):
            """
            Computes the Gaussian function.
            :param x: Point to evaluate the Gaussian for.
            :param ampl: Amplitude.
            :param center: Center.
            :param dev: Width.
            :param elev: Elevation above ground.
            :return: Value of the specified Gaussian at *x*
            """
            return ampl * np.exp(-(x - center) ** 2 / (2 * dev ** 2)) + elev

        # Initialising guess
        guess = [np.amax(self.counts[x_peak - 10 : x_peak + 10]), x_peak, 1, 100]

        # Extracting optimal parameters
        popt, pcov = sp.optimize.curve_fit(gaussian, self.channels[x_peak - 10 : x_peak + 10],
                                           self.counts[x_peak - 10 : x_peak + 10], guess)

        mean = popt[1]

        # Finding the n value for 99.7% confidence
        n = 3 * popt[2]

        x_data = self.channels[int(np.floor(mean - n))-1:int(np.ceil(mean + n))]
        y_data = self.counts[int(np.floor(mean - n))-1:int(np.ceil(mean + n))]

        # Initialising guess again
        guess = [np.amax(y_data), x_peak, 1, 1000]

        popt, pcov = sp.optimize.curve_fit(gaussian, x_data, y_data, guess)

        ampl = popt[0]
        dampl = np.sqrt(np.diag(pcov)[0])
        mean = popt[1]
        dmean = np.sqrt(np.diag(pcov)[1])
        stdev = popt[2]
        dstdev = np.sqrt(np.diag(pcov)[2])
        elev = popt[3]
        delev = np.sqrt(np.diag(pcov)[3])

        if show:
            x_gaussian = np.linspace(np.amax(x_data), np.amin(x_data), 200)
            y_gaussian = gaussian(x_gaussian, ampl, mean, stdev, elev)
            plt.figure()
            plt.plot(x_data, y_data, 'k.', markersize=1)
            plt.plot(x_gaussian, y_gaussian, 'r-', linewidth=0.5)
            plt.grid(True, linestyle='--', linewidth=0.2)

            plt.axhline(y=popt[3], color='g', linestyle='-', linewidth=0.35)

            plt.show()

        if mode == "Amplitude":
            return ampl, dampl

        elif mode == "Mean":
            return mean, dmean

        elif mode == "Variance":
            return stdev**2, 2*stdev*dstdev

        elif mode == "AllParams":
            return popt, pcov

        elif mode == "FWHM":
            FWHM = 2*np.sqrt(2*np.log(2))*stdev
            d_FWHM = 2*np.sqrt(2*np.log(2))*dstdev
            return FWHM, d_FWHM

        elif mode == "Area":
            trapezium = np.size(x_data) * int(round(0.5*(y_data[0]+y_data[np.size(y_data)-1])))
            return np.sum(y_data) - trapezium

        else:
            print("Choose a valid mode from on of the following options: Mean, Var, FWHM")
            return

    @staticmethod
    def auto_find_cal_factors(spectrum, energy_peaks, show_steps=True):
        """
        Calibrates the energy scale of the spectrum. Returns an array of energies.
        :param spectrum: object name or filename (if the input is a str) containing the spectrum data.
        :param energy_peaks: the position of peaks in terms of energies in increasing order, array.
        :param show_steps: a boolean of whether to plot the found peaks and fitted gaussians.
        :return: Scaling and shifting calibration parameters.
        """

        if type(spectrum) == str:
            cal = Spectrum()
            cal.read_from_csv(spectrum)
            spectrum = cal

        peaks = spectrum.find_peaks(show=show_steps)

        #  Checks if there is a mismatch between the number of input peaks and automatically found ones.
        if np.size(energy_peaks) != np.size(peaks):
            print("The number of theoretical peaks doesn't coincide with the number of found peaks.")
            return

        channel_peaks = np.array([])  # Array for peaks found using gaussian fitting.
        n = 10  # Why 10? WIP
        for i in peaks:
            peakmean, dpeakmean = spectrum.fit_gaussian(spectrum.channels[i-n:i+n], spectrum.counts[i-n:i+n],
                                                        "Mean", show=show_steps)
            channel_peaks = np.append(channel_peaks, peakmean)

        correction_factor = np.array([])  # Initialising an empty array for correction factor
        for i in range(np.size(energy_peaks)-1):
            energy_dif = energy_peaks[i + 1] - energy_peaks[i]
            channel_dif = channel_peaks[i + 1] - channel_peaks[i]
            correction_factor = np.append(correction_factor, energy_dif / channel_dif)
        correction_factor = np.mean(correction_factor)

        spectrum.energies = np.multiply(spectrum.channels, correction_factor)
        correction_shift = np.array([])  # Initialising an empty array for correction shift
        for i in range(np.size(energy_peaks)):
            shift = energy_peaks[i] - channel_peaks[i]*correction_factor
            correction_shift = np.append(correction_shift, shift)
        correction_shift = np.mean(correction_shift)
        return correction_factor, correction_shift

    def from_peaks_find_cal_factors (self, channel_peaks, energy_peaks, show_steps=True):

        if np.size(energy_peaks) != np.size(channel_peaks):
            print("The number of theoretical peaks doesn't coincide with the number of found peaks.")
            return

        peaks = channel_peaks
        channel_peaks = np.array([])  # Array for peaks found using gaussian fitting.
        n = 10  # Why 10? WIP
        for i in peaks:
            peakmean, dpeakmean = Spectrum.fit_gaussian(self.channels[i - n:i + n],
                                                        self.counts[i - n:i + n], "Mean", show=show_steps)
            channel_peaks = np.append(channel_peaks, peakmean)

        correction_factor = np.array([])  # Initialising an empty array for correction factor
        for i in range(np.size(energy_peaks) - 1):
            energy_dif = energy_peaks[i + 1] - energy_peaks[i]
            channel_dif = channel_peaks[i + 1] - channel_peaks[i]
            correction_factor = np.append(correction_factor, energy_dif / channel_dif)
        correction_factor = np.mean(correction_factor)

        self.energies = np.multiply(self.channels, correction_factor)
        correction_shift = np.array([])  # Initialising an empty array for correction shift
        for i in range(np.size(energy_peaks)):
            shift = energy_peaks[i] - channel_peaks[i] * correction_factor
            correction_shift = np.append(correction_shift, shift)
        correction_shift = np.mean(correction_shift)

        return correction_factor, correction_shift

    def calibrate(self, mode, correction_factor=None, correction_shift=None,
                  spectrum=None, energy_peaks=None, channel_peaks=None, show_steps=True):

        #  Automatic mode that find peaks by itself, and compares to the theoretical values.
        #  Required inputs:
        #       - Spectrum
        #       - energy_peaks
        if mode == "Auto":
            if (energy_peaks is None) or (spectrum is None):
                print("No energy peaks or spectrum given, can't find calibrating factors.")
                return

            correction_factor, correction_shift = Spectrum.auto_find_cal_factors(spectrum, energy_peaks, show_steps=show_steps)

        # Mode that starts from user-found peaks. Plots gaussian around the peak channel number,
        # compares to theoretical values.
        #  Required inputs:
        #       - channel_peaks
        #       - energy_peaks
        elif mode == "From_peaks":
            if (energy_peaks is None) or (channel_peaks is None):
                print("No channel or energy peaks given, can't find calibrating factors.")
                return

            if np.size(energy_peaks) != np.size(channel_peaks):
                print("The number of theoretical peaks doesn't coincide with the number of found peaks.")
                return

            correction_factor, correction_shift = self.from_peaks_find_cal_factors\
                (channel_peaks=channel_peaks, energy_peaks=energy_peaks)

        #  Simply applies user-found calibration factors.
        #  Required inputs:
        #       - correction_factor
        #       - correction_shift
        elif mode == "Apply_factors":
            if (correction_factor is None) or (correction_shift is None):
                print("Only one calibrating factor given, cannot calibrate.")
                return

        else:
            print("Choose a valid mode from on of the following options: Auto, From_peaks, Apply_factors")

        self.energies = np.add(np.multiply(self.channels, correction_factor), correction_shift)

    @staticmethod
    def reduce_background(spectrum, bkgd_spectrum):
        """
        Subtracts the background spectrum from the material one.
        Best used with the background spectrum taken right before or after the material spectrum was recorded.
        :param spectrum: Filename (if a string) or an object of the material spectrum
        :param bkgd_spectrum: Filename (if a string) or an object of the background spectrum
        :return:
        """

        if type(spectrum) == str:
            cal = Spectrum()
            cal.read_from_csv(spectrum)
            spectrum = cal

        if type(bkgd_spectrum) == str:
            cal = Spectrum()
            cal.read_from_csv(bkgd_spectrum)
            bkgd_spectrum = cal

        # Counts factor, comes from comparing the time both spectra were taken for.
        factor = spectrum.live_time/bkgd_spectrum.live_time

        new_spectrum = Spectrum()
        new_spectrum.__dict__.update(spectrum.__dict__)
        new_spectrum.counts = (spectrum.counts-bkgd_spectrum.counts*factor).astype(int)
        return new_spectrum

    @staticmethod
    def polynomial_calibration(channel_peaks, energy_peaks, show=True):

        #  Checks if there is a mismatch between the number of input peaks and automatically found ones.
        if np.size(energy_peaks) != np.size(channel_peaks):
            print("The number of theoretical peaks doesn't coincide with the number of found peaks.")
            return

        def quadratic(x, a, b, c):
            return a*x**2 + b*x + c

        # Initialising guess
        guess = [1, 1, 1]

        # Extracting optimal parameters
        popt, pcov = sp.optimize.curve_fit(quadratic, channel_peaks, energy_peaks, guess)

        if show:
            x = np.linspace(0, np.amax(channel_peaks)*1.1, 100)
            y = quadratic(x, popt[0], popt[1], popt[2])
            plt.figure()
            plt.title("Polynomial fitting for energy calibration.")
            plt.plot(channel_peaks, energy_peaks, "k.")
            plt.plot(x, y, "r--", linewidth=0.5)
            plt.xlabel("Channel")
            plt.ylabel("Energy, KeV")
            plt.xlim(0, np.amax(x))
            plt.ylim(0, np.amax(y))
            plt.grid(True, linestyle='--', linewidth=0.2)
            plt.show()

        Spectrum.calibration_a = popt[0]
        Spectrum.calibration_b = popt[1]
        Spectrum.calibration_c = popt[2]
        Spectrum.d_a = np.diag(pcov)[0]
        Spectrum.d_b = np.diag(pcov)[1]
        Spectrum.d_c = np.diag(pcov)[2]

    def apply_polynomial_calibration(self):
        # Finding each term separately.
        ax2 = np.multiply(np.multiply(self.channels, self.channels),
                          Spectrum.calibration_a)
        bx = np.multiply(self.channels, Spectrum.calibration_b)
        self.energies = np.add(np.add(ax2, bx), Spectrum.calibration_c)

        # Uncertainty in each channel is half a channel
        d_channel = 0.5
        d_ax2 = np.multiply(ax2, np.sqrt((2*d_channel/self.channels)**2
                            + (Spectrum.d_a/Spectrum.calibration_a)**2))

        d_bx = np.multiply(bx, np.sqrt((d_channel/self.channels)**2
                            + (Spectrum.d_b/Spectrum.calibration_b)**2))

        self.d_energies = np.sqrt(np.add(d_ax2, np.add(d_bx, Spectrum.d_c)))
        # AFTER TESTING OF CALIBRATION FROM C060 AND BI207,
        # THE UNCERTAINTY IN ENEGRY IS +-0.62KeV THROUGHOUT

    @staticmethod
    def channel_to_energy(channel):
        return Spectrum.calibration_a * channel ** 2 \
               + Spectrum.calibration_b * channel \
               + Spectrum.calibration_c
