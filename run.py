import find_gradient 
import find_lines 
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import sys

image_paths = []
count = 0


plot_fourier = False #Set to True for 2d fourier transform plot to shown
plot_line_find = False #Set to True for found lines plot to be shown
pixel_value_cutoff = 0.5 #This is the avg pixel value of the lines, below which lines will not be detected. 

image_paths.append('../Media/big_image_512/3.png')
image_paths.append('../Media/big_image_512/22.png')

image_paths.append('../Media/big_image_512/29.png')
image_paths.append('../Media/big_image_512/79.png')
image_paths.append('../Media/big_image_512/0.png')
image_paths.append('../Media/big_image_512/1.png')
# image_paths.append('../Media/0pos.png')
# image_paths.append('../Media/0.png')
image_paths.append('../Media/big_image_512/116.png')
image_paths.append('../Media/big_image_512/131.png')
image_paths.append('../Media/big_image_512/78.png')


halfway = round(len(image_paths)/2)



#find good estimation of gradient from the middle images
for i in range(halfway, halfway + 10):
	print(image_paths[i])
	img = plt.imread(image_paths[i]).astype(float)
	row_distance, source_angle = find_gradient.run_everything(img, plot_fourier)
	if round(source_angle) == 0 or round(source_angle) == -90 or round(source_angle) == 90 or round(source_angle) == 45 or round(source_angle) == -45:
		print('Bad angle:')
		print(source_angle)
	else:
		source_gradient = np.tan(np.deg2rad(source_angle)) 
		source_gradient, detected_lines = find_lines.run_everything(img, source_gradient, row_distance, plot_line_find, pixel_value_cutoff)
		break


print('selected gradient:')
print(source_gradient)

# sys.exit()

for image in image_paths: #Loop through images provided in image_paths
	print('**--**')
	print(image)
	img = plt.imread(image).astype(float)
	
	row_distance, row_angle = find_gradient.run_everything(img, plot_fourier)

	#checking if the detected row_angle is ok
	if round(row_angle) == 0 or round(row_angle) == -90 or round(row_angle) == 90 or round(row_angle) == 45 or round(row_angle) == -45:
		print('detected bad angle, using source_angle')
		gradient = source_gradient
	else:
		gradient = np.tan(np.deg2rad(row_angle)) 
	
	if count > 0 and improved_gradient and abs(improved_gradient - gradient) < 0.5:

		print('running with improved_gradient:')
		print(improved_gradient)
		improved_gradient, detected_lines = find_lines.run_everything(img, improved_gradient, row_distance, plot_line_find, pixel_value_cutoff)
	else:
		improved_gradient, detected_lines = find_lines.run_everything(img, gradient, row_distance, plot_line_find, pixel_value_cutoff)

	count += 1



#Results of bad images:

# ../Media/big_image_512/2.png => row_angle = -90 

# ../Media/big_image_512/7.png => row_angle = -0

# ../Media/big_image_512/29.png => row_angle = 90 


