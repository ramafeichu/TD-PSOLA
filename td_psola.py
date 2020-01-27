"""
Based on https://github.com/sannawag/TD-PSOLA by Sanna Wager (9/18/19)

Authors:
Ramiro Feichubuinm
Ramiro Merello


Bibliography:
-Rudresh S, Vasisht A, Vijayan K and Seelamantula CS (2018), "Epoch-synchronous overlap-add (ESOLA) for time-and pitch-scale modification of speech signals", arXiv preprint arXiv:1801.06492.
-Zölzer U (2011), "DAFX: digital audio effects" John Wiley & Sons.
-von dem Knesebeck A and Zölzer U (2010), "Comparison of pitch trackers for real-time guitar effects", In Proc. 13th Int. Conf. Digital Audio Effects.
-Peeters G (2006), "Music pitch representation by periodicity measures based on combined temporal and spectral representations", In 2006 IEEE International Conference on Acoustics Speech and Signal Processing Proceedings. Vol. 5, pp. V-V.
-Kortekaas RW and Kohlrausch A (1997), "Psychoacoustical evaluation of the pitch-synchronous overlap-and-add speech-waveform manipulation technique using single-formant stimuli", The Journal of the Acoustical Society of America. Vol. 101(4), pp. 2202-2213. ASA.
-Moulines E and Laroche J (1995), "Non-parametric techniques for pitch-scale and time-scale modification of speech", Speech communication. Vol. 16(2), pp. 175-205. Elsevier.
-Moulines E and Laroche J (1995), "Non-parametric techniques for pitch-scale and time-scale modification of speech", Speech communication. Vol. 16(2), pp. 175-205. Elsevier.
-Bristow-Johnson R (1993), "A detailed analysis of a time-domain formant-corrected pitch-shifting algorithm", In Audio Engineering Society Convention 95.
-Valbret H, Moulines E and Tubach J-P (1992), "Voice transformation using PSOLA technique", Speech communication. Vol. 11(2-3), pp. 175-187. Elsevier.
-Moulines E and Charpentier F (1990), "Pitch-synchronous waveform processing techniques for text-to-speech synthesis using diphones", Speech communication. Vol. 9(5-6), pp. 453-467. Elsevier.
-Charpentier F and Stella M (1986), "Diphone synthesis using an overlap-add technique for speech waveforms concatenation", In ICASSP'86. IEEE International Conference on Acoustics, Speech, and Signal Processing. Vol. 11, pp. 2015-2018.

"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import parselmouth


def get_fundamental(signal, fs):
    """
    This function estimates the fundamental frequency of the signal
    Input:
        -:param signal: original signal in the time-domain
        -:param fs: sample rate

    Return:
        -:return: fundamental frequency (pitch)
    """
    peaks = find_peaks(signal, fs)
    periods = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    period = np.mean(periods)
    f0 = fs/period
    return f0


# from https://parselmouth.readthedocs.io/en/stable/examples/plotting.html
def draw_pitch(pitch, label):
    """
    This function draws the estimated pitch

    Input:
        -:param pitch: array returned by Sound.to_pitch()
        -:param label: plot label

    Return: NO RETURN VAL
        -:return: -
    """
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2, label=label)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("Frecuencia Fundamental [Hz]")
    plt.xlabel("Tiempo [s]")


def stretch_time(signal, fs, t_ratio):
    """
    This function caries out the time stretching of the PSOLA algorithm

    Input:
        -:param signal: original signal in the time-domain
        -:param fs: sample rate of signal
        -:param t_ratio: ratio by which the time will be stretched

    Return:
        -:return: time-stretched signal
    """
    peaks = find_peaks(signal, fs)
    new_signal = tsola(signal, peaks, t_ratio)
    return new_signal


def shift_pitch(signal, fs, f_ratio):
    """
    This function caries out the pitch shifting of the PSOLA algorithm

    Input:
        -:param signal: original signal in the time-domain
        -:param fs: sample rate of signal
        -:param f_ratio: ratio by which the frequency will be shifted

    Return:
        -:return: pitch-shifted signal
    """
    peaks = find_peaks(signal, fs)
    new_signal = psola(signal, peaks, f_ratio)
    return new_signal


def find_peaks(signal, fs, max_hz=950, min_hz=75, analysis_win_ms=40, max_change=1.3, min_change=0.7):
    """
    This functions finds indexes of peaks in time-domain signal

    Input:
        -:param max_hz: maximum measured fundamental frequency
        -:param min_hz: minimum measured fundamental frequency
        -:param analysis_win_ms: window size used for autocorrelation analysis
        -:param max_change: restrict periodicity to not increase by more than this ratio from the mean
        -:param min_change: restrict periodicity to not decrease by more than this ratio from the mean

    Return:
        -:return: peak indexes
    """
    N = len(signal)
    min_period = fs // max_hz
    max_period = fs // min_hz

    # compute pitch periodicity
    sequence = int(analysis_win_ms / 1000 * fs)  # sequence length in samples

    # Acá saqué una parte de código que evitaba tener errores de octava en la detección,
    # pero a su vez impedía que haya variaciones mayores en el pitch.

    offset = 0  # current sample offset
    periods = []  # period length of each analysis sequence

    while offset < N:
        fourier = fft(signal[offset: offset + sequence])
        fourier[0] = 0  # remove DC component
        autoc = ifft(fourier * np.conj(fourier)).real
        autoc_peak = min_period + np.argmax(autoc[min_period: max_period])
        periods.append(autoc_peak)
        offset += sequence

    # find the peaks
    peaks = [np.argmax(signal[:int(periods[0]*1.1)])]
    while True:
        prev = peaks[-1]
        idx = prev // sequence  # current autocorrelation analysis window
        if prev + int(periods[idx] * max_change) >= N:
            break
        # find maximum near expected location
        peaks.append(prev + int(periods[idx] * min_change) +
                np.argmax(signal[prev + int(periods[idx] * min_change): prev + int(periods[idx] * max_change)]))
    return np.array(peaks)


def psola(signal, peaks, f_ratio):
    """
    Time-Domain Pitch Synchronous Overlap and Add

    Input:
        -:param signal: original time-domain signal
        -:param peaks: time-domain signal peak indices
        -:param f_ratio: pitch shift ratio

    Return:
        -:return: pitch-shifted signal
    """
    N = len(signal)

    new_signal = np.zeros(N)
    new_peaks_ref = np.linspace(0, len(peaks) - 1, len(peaks) * f_ratio)
    new_peaks = np.zeros(len(new_peaks_ref)).astype(int)

    # When creating the new peaks calculates the weighted sum of the original adjacent peaks
    for i in range(len(new_peaks)):
        weight = new_peaks_ref[i] % 1
        left = np.floor(new_peaks_ref[i]).astype(int)
        right = np.ceil(new_peaks_ref[i]).astype(int)
        new_peaks[i] = int(peaks[left] * (1 - weight) + peaks[right] * weight)

    # Overlap-and-add:
    for j in range(len(new_peaks)):
        # find the corresponding old peak index
        i = np.argmin(np.abs(peaks - new_peaks[j]))
        # get the distances to adjacent peaks
        P1 = [new_peaks[j] if j == 0 else new_peaks[j] - new_peaks[j-1],
              N - 1 - new_peaks[j] if j == len(new_peaks) - 1 else new_peaks[j+1] - new_peaks[j]]
        # edge case truncation
        if peaks[i] - P1[0] < 0:
            P1[0] = peaks[i]
        if peaks[i] + P1[1] > N - 1:
            P1[1] = N - 1 - peaks[i]
        # Windowing
        window = list(np.hanning(P1[0] + P1[1]))
        # window = list(np.hamming(P1[0] + P1[1]))

        # center window from original signal at the new peak
        new_signal[new_peaks[j] - P1[0]: new_peaks[j] + P1[1]] += window * signal[peaks[i] - P1[0]: peaks[i] + P1[1]]

    return new_signal


def tsola(signal, peaks, t_ratio):
    """
        Time-Domain Pitch Synchronous Overlap and Add

        Input:
            -:param signal: original time-domain signal
            -:param peaks: time-domain signal peak indices
            -:param t_ratio: time stretch ratio

        Return:
            -:return: time-stretched signal
        """
    N = len(signal)
    new_signal = np.zeros(int(round(N * t_ratio)))
    n = len(new_signal)
    new_peaks_ref = np.round(np.linspace(0, len(peaks) - 1, round(len(peaks) * t_ratio)))
    new_peaks = np.zeros(len(new_peaks_ref)).astype(int)

    periods = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    periods.insert(0, 0)

    new_peaks[0] = peaks[0]
    for i in range(1, len(new_peaks)):
        new_peaks[i] = new_peaks[i-1] + periods[int(new_peaks_ref[i])]
    new_peaks = np.array(new_peaks)

    # Overlap-and-add:
    for j in range(len(new_peaks)):
        # find the corresponding old peak index
        i = int(new_peaks_ref[j])

        # get the distances to adjacent peaks
        P1 = [new_peaks[j] if j == 0 else new_peaks[j] - new_peaks[j - 1],
              N - 1 - new_peaks[j] if j == len(new_peaks) - 1 else new_peaks[j + 1] - new_peaks[j]]

        # edge case truncation
        if new_peaks[i] - P1[0] < 0:
            P1[0] = new_peaks[i]
        if new_peaks[i] + P1[1] > n - 1:
            P1[1] = n - 1 - new_peaks[i]

        # Windowing
        window = list(np.hanning(P1[0] + P1[1]))
        # window = list(np.hamming(P1[0] + P1[1]))

        # center window from original signal at the new peak
        new_signal[new_peaks[j] - P1[0]: new_peaks[j] + P1[1]] += window * signal[peaks[i] - P1[0]: peaks[i] + P1[1]]
    return new_signal


if __name__=="__main__":

    orig_signal, fs = librosa.load("indian.aif", sr=44100)

    """PITCH SHIFTING"""

    f_ratio = 1.3

    pitch_shifted = shift_pitch(orig_signal, fs, f_ratio)

    """TIME STRETCHING"""
    # Utilizar t_ratio en el tramo [1.1 ; 1.9]. Fuera de este rango obtuvimos problemas que no pudimos solucionar.

    t_ratio = 1.8

    time_stretched = stretch_time(orig_signal, fs, t_ratio)

    sf.write("pitch_shifted_{:01.2f}.wav".format(f_ratio), pitch_shifted, fs)
    sf.write("time_stretched_{:01.2f}.wav".format(t_ratio), time_stretched, fs)
