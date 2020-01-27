
#Load audio
orig_signal, fs = librosa.load("U11.wav", sr=44100)

for i in range(0, 10):
    new_signal = shift_pitch(orig_signal, fs, i*0.1)
    librosa.output.write_wav("female_scale_transposed_{:01.2f}.wav".format(i*0.1), new_signal, fs)

