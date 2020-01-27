#Esta es la original a aprox 440Hz
orig_signal, fs = librosa.load("U11.wav", sr=44100)

f_ratio = 0.882

#Lo llevo a 396Hz
new_signal = shift_pitch(orig_signal, fs, f_ratio)
new_signal = new_signal[150000:155000]
new_signal = new_signal / np.max(new_signal)

#Esta es la original a aprox 396Hz
_signal, fs = librosa.load("U2.wav", sr=44100)
_signal = _signal[150000:155000]
_signal = _signal / np.max(_signal)

Y1 = np.abs(fft(new_signal)/len(new_signal)) ** 2
freq1 = fftfreq(len(new_signal), 1 / 44100)

Y2 = np.abs(fft(_signal)/len(_signal)) ** 2
freq2 = fftfreq(len(_signal), 1 / 44100)

plt.semilogx(freq1, Y1, label="PSOLA 396Hz")
plt.semilogx(freq2, Y2, label="Original 396Hz")
plt.xlabel("Frecuencia(Hz)")
plt.ylabel("Magnitud")
plt.legend()
plt.show()