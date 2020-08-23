from PIL import Image, ImageFilter
import numpy as np
import glob
import os
from perlin2d import generate_perlin_noise_2d, generate_fractal_noise_2d
import re


def generate_simulated(image, res, alpha=0.5, octaves=1, lacunarity=2):
# This method gets a PIL image, alpha (0-1) and perlin parameters as input 
#and returns the combined weighted noisy PIL image
	width, height = image.size
	# A calaculation of the matching width and height of the image, according to the perlin image size
	mod_width = res[0]*np.power(lacunarity,octaves-1)
	mod_height = res[1]*np.power(lacunarity,octaves-1)
	new_image = image
	
	# Check if width/height of the original image fits. if not - resize original image
	if width%mod_width != 0 and height%mod_height != 0:
		# If width/height of the original image is too small, resize it to the next multiply of perlin size
		if mod_width >= width or mod_height >= height:
			new_image = image.resize((2*width-((2*width)%mod_width), 2*height-((2*height)%mod_height)))
		else:
			new_image = image.resize((width-(width%mod_width), height-(height%mod_height)))
	
	if width%mod_width != 0 and height%mod_height == 0:
		if mod_width >= width:
			new_image = image.resize((2*width-((2*width)%mod_width), height))
		else:
			new_image = image.resize((width-(width%mod_width), height))
	
	if width%mod_width == 0 and height%mod_height != 0:
		if mod_height >= height:
			new_image = image.resize((width, 2*height-((2*height)%mod_height)))
		else:
			new_image = image.resize((width, height-(height%mod_height)))
	
	new_width, new_height = new_image.size
	# Generate perlin image with the given "generate_fractal_noise_2d" method and image's new shape
	np_perlin = generate_fractal_noise_2d((new_width,new_height), (8,8), octaves)
	np_image = np.asarray(new_image).astype('uint8')
	# Duplicate perlin to 3 channels to fit RGB image
	np_perlin = np.repeat(np_perlin[:, :, np.newaxis], 3, axis=2)
	# Fit perlin to image's shape
	np_perlin = np.transpose(np_perlin, (1, 0, 2))
	# Convert perlin image pixels from values in range [-2, 2] to values between 0 and 255
	np_perlin = ((np_perlin + 2) * (1/4 * 255)).astype('uint8')
	
	print('perlin shape: ', np_perlin.shape)
	print('resized image shape: ', np_image.shape)
	
	# Check if original image has not 3 channels (like gray-scale) and duplicate it if necessary 
	if np_image.ndim != 3:
		np_image = np.repeat(np_image[:, :, np.newaxis], 3, axis=2)
	
	# Apply a weighted sum of perlin and image
	np_combined = alpha*np_image + (1-alpha)*np_perlin
	combined = Image.fromarray(np_combined.astype('uint8'))
	return combined, np_perlin

def convert_images(res=(8,8), alpha=0.5, octaves=1, lacunarity=2):
	# This method gets alpha (0-1) and perlin parameters as input 
	#and creates hazy images
	directory_to_check = "/data/image_processing/data/ADE_new/images" # Which directory do you want to start with?
	directory_train = "/data/image_processing/data/ADE_new/images/training"
	#directory_test = "/data/image_processing/data/ADEChallengeData2016/images/testing"
	directory_val = "/data/image_processing/data/ADE_new/images/validation"

	# Create a dict for all paths of original images to walk through
	directories = [os.path.abspath(x[0]) for x in os.walk(directory_to_check)]
	directories.remove(os.path.abspath(directory_to_check)) # Remove parent directory

	for i in directories:
		os.chdir(i)  # Change working Directory
		for image_file in glob.iglob('./*.jpg'):
			filename = os.path.basename(image_file)  # Get original image name
			filename2 = os.path.splitext(filename)[0]  # Cut its name to basic version
			res_im = re.findall("(\w+)_ADE", filename2)  # Save name's beginning
			if not not res_im:  # If already transformed - continue
				continue
			print(filename2)
			im = Image.open(image_file)
			# Apply transformation on image
			combined_image, np_perlin = generate_simulated(im, res, alpha, octaves)
			# Save the new image
			combined_image.save('new_'+filename)
			combined_image.close()
			im.close()
		
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	alpha = 0.5
	octaves = 5
	res = (8,8)
	
	# Convert all images
	convert_images(res, alpha, octaves)
	
	"""
	PATH = '/data/image_processing/data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg'
	im = Image.open(PATH)
	combined_image, np_perlin = generate_simulated(im, res, alpha, octaves)

	np.random.seed(0)
	plt.imshow(combined_image)
	plt.axis('off')
	plt.savefig('/data/image_processing/data/combined_image.jpg')
	
	#perlin_3d = Image.fromarray(np_perlin.astype(np.uint8))
	np.random.seed(0)
	plt.imshow(np_perlin, cmap='gray', interpolation='lanczos')
	plt.axis('off')
	plt.savefig('/data/image_processing/data/perlin_3d.jpg')
	"""
