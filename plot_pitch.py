orig_signal, fs = librosa.load("indian.aif", sr=44100)

f_ratio1 = 0.7
f_ratio2 = 1.3

new_signal1 = shift_pitch(orig_signal, fs, f_ratio1)
new_signal2 = shift_pitch(orig_signal, fs, f_ratio2)

# librosa.output.write_wav("indian_{:01.2f}.wav".format(f_ratio1), new_signal1, fs)
sf.write("indian_{:01.2f}.wav".format(f_ratio1), new_signal1, fs)
sf.write("indian_{:01.2f}.wav".format(f_ratio2), new_signal2, fs)
# librosa.output.write_wav("indian_{:01.2f}.wav".format(f_ratio2), new_signal2, fs)

snd1 = parselmouth.Sound("indian.aif")
snd2 = parselmouth.Sound("indian_{:01.2f}.wav".format(f_ratio1))
snd3 = parselmouth.Sound("indian_{:01.2f}.wav".format(f_ratio2))

pitch1 = snd1.to_pitch()
pitch2 = snd2.to_pitch()
pitch3 = snd3.to_pitch()

draw_pitch(pitch1, 'Original')
draw_pitch(pitch2, 'PSOLA f_ratio=0.7')
draw_pitch(pitch3, 'PSOLA f_ratio=1.3')
plt.legend()
plt.show()