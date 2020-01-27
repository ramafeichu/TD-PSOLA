# Load audio
orig_signal, fs = librosa.load("U11.wav", sr=44100)
#
# #Pitch shift amount as a ratio
f_ratio = 0.882
#
# #Shift pitch
new_signal = shift_pitch(orig_signal, fs, f_ratio)
new_signal = new_signal[150000:155000]
new_signal = new_signal / np.max(new_signal)

# Write to disk
# librosa.output.write_wav("result_{:01.2f}.wav".format(f_ratio), new_signal, fs)

_signal, fs = librosa.load("U2.wav", sr=44100)
_signal = _signal[150000:155000]
_signal = _signal / np.max(_signal)

plt.plot(_signal, label='Original 396Hz')
plt.plot(new_signal, label='PSOLA 396Hz')
plt.xlabel("Samples")
plt.ylabel("Amplitud Normalizada")
plt.legend()
plt.show()