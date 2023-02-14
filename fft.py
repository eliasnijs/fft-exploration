#!/bin/python3

import pathlib
import sounddevice
import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from typing import List

# --------------------------- FFT utilities ------------------------------------

#vraag 1
def maak_signalen(compositie: List[list], sample_rate: int, nr_sample_points: int) -> list:
	signals = np.zeros(nr_sample_points)
	samplepoints = np.linspace(0, nr_sample_points / sample_rate,
			 nr_sample_points)
	for fq, fs, amp in compositie:
		signals += amp * np.sin(2 * np.pi * fq * samplepoints + fs)
	return signals.tolist()

#vraag 2
def fftfreq(n, d):
	f = np.arange(0, n/2 + 1) / (n * d)
	return np.concatenate((f, -f[-2:0:-1]))

def fft(signaal: np.ndarray, sample_rate: int) -> (np.ndarray, np.ndarray):
	n = len(signaal)

	if (n <= 1):
		return np.array([]), signaal

	_, ft_even = fft(signaal[::2], sample_rate)
	_, ft_odd = fft(signaal[1::2], sample_rate)

	w = np.exp(-2j*np.pi*np.arange(n) / n)
	X = np.concatenate((ft_even + w[:n//2]*ft_odd,  ft_even + w[n//2:]*ft_odd))
	freq = fftfreq(n, 1/sample_rate)

	return freq, X

# --------------------------- animation of FFT --------------------------------

CHUNK = 1024*4
# Hz for music -> might be 48000 as well
RATE = 44100

FREQ_BAND = [500, 10e5]

def open_audio(file_path: pathlib.Path):
	global RATE, CHUNK
	RATE, signal = wf.read(file_path)
	stream = sounddevice.OutputStream(
                samplerate=RATE,
                blocksize=CHUNK,
                channels = len(signal.shape),
                dtype = signal.dtype
                )
	stream.start()

	fig, ax, line = set_up_eq()

	for frame_nr in range(signal.shape[0] // CHUNK):
                music_chunk = signal[CHUNK*frame_nr:CHUNK*(frame_nr+1)]
                audio, data_fft = process_geluid(music_chunk, FREQ_BAND[0], FREQ_BAND[1])
                stream.write(np.ascontiguousarray(audio,dtype=audio.dtype))
                if len(data_fft.shape) == 1:
                        # send data to plot next figure in eq
                        plot_equalizer(data_fft, fig, ax, line)
                else:
                        # send data to plot next figure in eq
                        plot_equalizer(data_fft[:,0], fig, ax, line)

	stream.stop()
	stream.close()

freq = None
def set_up_eq():
	global freq, CHUNK, RATE
	fig, ax = plt.subplots(figsize=(10,5))
	freq = np.fft.fftfreq(CHUNK, 1/RATE)[:CHUNK//2 + 1]
	line, = ax.semilogx(freq, np.zeros(CHUNK // 2 + 1))
	ax.set_xlim(20, int(RATE / 2)) # going from 20 Hz to Rate/2 Hz
	ax.set_ylim(0,255)
	ax.set_xlabel("freq (Hz)")
	ax.set_ylabel("Amplitude")
	plt.show(block=False)
	return fig, ax, line

prev_data = np.zeros(CHUNK // 2 + 1)
alpha = 0.5
def plot_equalizer(data_fft, fig, ax, line):
	global CHUNK, prev_data, alpha
	fft_vals = data_fft[:CHUNK//2 + 1]
	s_mag = np.abs(fft_vals) * 2 / CHUNK
	# -> should be rescalled between 0 and 255, but bit small so do times 4
	freq_data = s_mag * 4 * 255 / 32768
	to_plot_val = freq_data
	# smoothing over time
	prev_data = alpha * to_plot_val + (1 - alpha) * prev_data
	# there may be no -inf in plot -> otherwise will not plot
	prev_data = np.where(prev_data == -np.inf, 0, prev_data)
	line.set_ydata(prev_data)
	try:
		ax.draw_artist(ax.patch)
		ax.draw_artist(line)
		fig.canvas.flush_events()
	except Exception as e:
		print("error: ", e)
		fig.canvas.draw()
		fig.canvas.flush_events()

win = np.hamming(CHUNK)
#vraag 4 & 5
def process_geluid(music_chunk: np.ndarray, start_freq: float, end_freq: float) -> (np.ndarray, np.ndarray):
	global win, freq, CHUNK, RATE

	fltr = np.fft.fftfreq(CHUNK, 1/RATE)
	fltr = ((fltr >= start_freq) & (fltr < end_freq)) | ((fltr <= -start_freq) & (fltr > -end_freq))

	fft = np.fft.fft(music_chunk)
	if len(music_chunk.shape) == 1:
		fft[fltr] = 0
		fft_plt = np.fft.fft(music_chunk * win)
	else:
		fft[:,fltr] = 0
		fft_plt = np.fft.fft(music_chunk[:,0] * win)
	fft_plt[fltr] = 0
	fft = np.fft.ifft(fft)

	return np.int16(fft), fft_plt

# --------------------------- plot utilities -----------------------------------

def plt_to_img(path, title, xname, yname, ax):
	ax.set_axisbelow(True)
	ax.yaxis.grid(color='silver')
	plt.subplots_adjust(bottom=0.15)
	plt.xlabel(xname)
	plt.ylabel(yname)
	plt.title(title)
	plt.legend()
	plt.savefig(f"{path}.eps", format='eps')
	plt.clf()


# --------------------------- tests --------------------------------------------
def test_vraag_1():
	compositie = [[1, 0, 1]]
	sample_rate = 100
	nr_sample_points = 300

	signal = maak_signalen(compositie, sample_rate, nr_sample_points)

	plt.plot(np.linspace(0, nr_sample_points / sample_rate, nr_sample_points), signal)
	plt.show()

def test_vraag_2():
	freq, fftr = fft(np.array([1, 1, -1, -1, -1, 1, 1, -1]), 100)
	print(f"           {np.round(fftr)}")
	print(f"Should be: [ 0.+0.j  2.+2.j -0.-4.j  2.-2.j  0.+0.j  2.+2.j  0.+4.j  2.-2.j]")

def test_vraag_4():
	open_audio("music.wav")

# --------------------------- report -------------------------------------------

def vraag_3_helper(name, composition, sample_rate, timeunits):
	nr_sample_points = timeunits*sample_rate
	signal = maak_signalen(composition, sample_rate, nr_sample_points)
	freq, X = fft(signal, sample_rate)
	X2 = np.fft.fft(signal)
	print(np.around(X2 - X, 3))
	fig,ax = plt.subplots()
	plt.plot(np.linspace(0, nr_sample_points / sample_rate, nr_sample_points), signal, label='signal')
	plt_to_img(f'report/images/3{name}_timedomain', 'Signaal in het tijdsdomein', 'Tijd (s)', 'Amplitude', ax)
	fig,ax = plt.subplots()
	plt.plot(freq, np.abs(X), label='signal')
	plt_to_img(f'report/images/3{name}_freqdomain', 'Signaal in het frequentiedomein', 'Frequentie (Hz)', 'Amplitude', ax)

def vraag_3_b():
	composition = [[2, 0, 1]]
	vraag_3_helper("b", composition, 32, 8)

def vraag_3_c():
	composition = [[2, 0, 1], [3, np.pi/2, 1], [7,0,1]]
	vraag_3_helper("c", composition, 32, 8)

def vraag_3_d():
	composition = [[2, 0, 1], [3, np.pi/2, 1], [7,0,1]]
	vraag_3_helper("d", composition, 16, 8)

def vraag_3_e():
	composition = [[2, 0, 1], [3, np.pi/2, 1], [7,0,1]]
	vraag_3_helper("e", composition, 8, 8)

def vraag_3_test():
	composition = [[1,0,1]]
	vraag_3_helper("test", composition, 32, 8)
	plt.show()

def vraag_4():
	# open_audio("audio/toonA.wav")
	# open_audio("audio/toonB.wav")
	open_audio("audio/toonC.wav")

def vraag_5():
	open_audio("...")

# --------------------------- main ---------------------------------------------

def main():
	vraag_3_test()
	return

if __name__ == "__main__":
    main()



