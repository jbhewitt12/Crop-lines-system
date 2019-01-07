# From https://www.scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html

import numpy as np
import matplotlib.pyplot as plt
import scipy 
from scipy import fftpack
from scipy import interpolate
import sys

# Reading the .tif this way produces an array of all 0's with the last row as 255 (doesn't work)
# img = plt.imread('nvdi_float.tif').astype(float)

def get_fourier_values(img):

	FFTData = np.fft.fft2(img) #Here we find the Fourier Transform of the image
	FreqCompRows = np.fft.fftfreq(FFTData.shape[0]) #frequency in 1/pixels 
	FreqCompCols = np.fft.fftfreq(FFTData.shape[1])

	shiftrows = fftpack.fftshift( FreqCompRows ) # shift so that low spatial frequencies are in the center.
	shiftcols = fftpack.fftshift( FreqCompCols )

	# print('rows:')
	# print(shiftrows)
	# print('cols:')
	# print(shiftcols)

	return shiftrows, shiftcols

def get_shifted_fourier_chart(img):
	fourier = fftpack.fft2(img)
	fourier_shifted = fftpack.fftshift( fourier ) # shift so that low spatial frequencies are in the center.
	fourier_abs = np.abs(fourier_shifted) #This array that we use is the absolute value of the shifted fourier transform

	return fourier_shifted, fourier_abs

def get_row_values(fourier_abs, shiftrows, shiftcols):
	# print('shape')
	# print(fourier_abs.shape)

	# Here we set the center point to 0 and then find the highest value (which should be the dot to the right)
	#find the index of the highest value point
	indices = np.unravel_index(np.argmax(fourier_abs, axis=None), fourier_abs.shape) 
	# print('index of max value:')
	# print(indices)

	fourier_abs[indices[0]][indices[1]] = 0 #set the max value to 0 as this middle point is not useful (note that the index of this point is the middle of the 2D array)

	indices = np.unravel_index(np.argmax(fourier_abs, axis=None), fourier_abs.shape)
	# print('new index of max value:')
	# print(indices)

	# print('value at new index:')
	# print(fourier_abs[indices[0]][indices[1]])

	row_frequency = shiftrows[indices[0]]
	column_frequency = shiftcols[indices[1]]

	frequency = np.sqrt(row_frequency*row_frequency + column_frequency*column_frequency)
	row_distance = 1/frequency

	# print(1/shiftrows[indices[0]])
	# print(1/shiftcols[indices[1]])

	# print("pixels between rows:")
	# print(row_distance) #number of pixels between rows 

	row_angle = np.arctan(column_frequency/row_frequency)*(180/np.pi) #degrees of the row_angle 

	print("row_angle:")
	print(row_angle)

	gradient = np.tan(np.deg2rad(row_angle)) 
	# print('gradient:')
	# print(gradient)

	return row_distance, row_angle

def plot_fourier(fourier_shifted, shiftrows, shiftcols):
	# These lines display the fourier plot
	plt.figure()
	plt.imshow(np.log(np.abs(fourier_shifted)), extent = (shiftcols[0], shiftcols[-1], shiftrows[0], shiftrows[-1]))
	plt.title('Fourier transform')

def plot_img(img, title):
	plt.figure()
	plt.imshow(img)
	plt.colorbar()
	plt.title(title)

# def zvalues_attempt(img):
# 	# construct interpolation function
# 	# (assuming your data is in the 2-d array "data")
# 	x = np.arange(img.shape[1])
# 	y = np.arange(img.shape[0])
# 	f = scipy.interpolate.interp2d(x, y, img)

# 	# extract values on line from r1, c1 to r2, c2
# 	r1 = 0
# 	c1 = 0
# 	r2 = 300
# 	c2 = 300
# 	num_points = 100
# 	xvalues = np.linspace(c1, c2, num_points)
# 	yvalues = np.linspace(r1, r2, num_points)
# 	zvalues = f(xvalues, yvalues)
# 	print('zvalues:')
# 	print(zvalues)
# 	print('zvalues shape:')
# 	print(zvalues.shape)

# 	return zvalues

def run_everything(img, plot):

	shiftrows, shiftcols = get_fourier_values(img)

	fourier_shifted, fourier_abs = get_shifted_fourier_chart(img)

	row_distance, row_angle = get_row_values(fourier_abs, shiftrows, shiftcols)

	if plot:

		plot_fourier(fourier_shifted, shiftrows, shiftcols)
		plt.show()

	# zvalues = zvalues_attempt(img)
	
	
	return row_distance, row_angle
	# plot_img(zvalues, 'zvalues')


def main_run():

	# We want the images to be square and a power of 2 (256 x 256) or (512 x 512)
	img = plt.imread('../Media/cropped512/0.png').astype(float)

	# print('image shape:')
	# print(img.shape)



	shiftrows, shiftcols = get_fourier_values(img)

	fourier_shifted, fourier_abs = get_shifted_fourier_chart(img)

	row_distance, row_angle = get_row_values(fourier_abs, shiftrows, shiftcols)

	plot_fourier(fourier_shifted, shiftrows, shiftcols)

	# zvalues = zvalues_attempt(img)

	plot_img(img, 'lines')
	# plot_img(zvalues, 'zvalues')


# ----- MAIN -----
if __name__ == "__main__":
	main_run()
	plt.show()


