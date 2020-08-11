from PIL import Image, ImageFilter
import numpy as np
import glob
import os
from perlin2d import generate_perlin_noise_2d, generate_fractal_noise_2d


def generate_simulated(image, res, alpha=0.5, octaves=1, lacunarity=2):
# This method gets a PIL image, alpha (0-1) and perlin noise as input 
#and returns the combined weighted noisy PIL image
	width, height = image.size
	mod_width = res[0]*np.power(lacunarity,octaves-1)
	mod_height = res[1]*np.power(lacunarity,octaves-1)
	new_image = image
	
	if width%mod_width != 0 and height%mod_height != 0:
		if (width%mod_width) >= width or height%mod_height >= height:
			new_image = image.resize((2*width-(width%mod_width), 2*height-(height%mod_height)))
		else:
			new_image = image.resize((width-(width%mod_width), height-(height%mod_height)))
	
	if width%mod_width != 0 and height%mod_height == 0:
		if (width%mod_width) >= width:
			new_image = image.resize((2*width-(width%mod_width), height))
		else:
			new_image = image.resize((width-(width%mod_width), height))
	
	if width%mod_width == 0 and height%mod_height != 0:
		if height%mod_height >= height:
			new_image = image.resize((width, 2*height-(height%mod_height)))
		else:
			new_image = image.resize((width, height-(height%mod_height)))
	
	new_width, new_height = new_image.size
	np_perlin = generate_fractal_noise_2d((new_width,new_height), (8,8), octaves)
	np_image = np.asarray(new_image).astype('uint8')
	np_perlin = np.repeat(np_perlin[:, :, np.newaxis], 3, axis=2)
	np_perlin = np.transpose(np_perlin, (1, 0, 2))
	# Convert perlin image pixels to values between 0 and 255
	np_perlin = ((np_perlin + 2) * (1/4 * 255)).astype('uint8')
	
	#pil_perlin = Image.fromarray(np_perlin.astype('uint8'))
	# Remove salt&pepper noise from perlin image
	#pil_perlin = pil_perlin.filter(ImageFilter.MedianFilter(size = 15))
	#np_perlin = np.asarray(pil_perlin).astype('uint8')
	
	print('perlin shape: ', np_perlin.shape)
	print('resized image shape: ', np_image.shape)
	#np_combined = np.zeros(np_image.shape)
	#for layer in range(np_image.shape[2]):
		#np_combined[:,:,layer] = alpha*np_image[:,:,layer] + (1-alpha)*np_perlin
	np_combined = alpha*np_image + (1-alpha)*np_perlin
	print(np.amax(np_perlin), np.amin(np_perlin))
	combined = Image.fromarray(np_combined.astype('uint8'))
	return combined, np_perlin

def convert_images(res=(8,8), alpha=0.5, octaves=1, lacunarity=2):
	directory_to_check = "/data/image_processing/data/ADEChallengeData2016/images" # Which directory do you want to start with?
	directory_train = "/data/image_processing/data/ADEChallengeData2016/images/training"
	#directory_test = "/data/image_processing/data/ADEChallengeData2016/images/testing"
	directory_val = "/data/image_processing/data/ADEChallengeData2016/images/validation"

	directories = [os.path.abspath(x[0]) for x in os.walk(directory_to_check)]
	directories.remove(os.path.abspath(directory_to_check)) # Remove parent directory

	for i in directories:
		os.chdir(i)  # Change working Directory
		for image_file in glob.iglob('./*.jpg'):
			if os.path.getmtime(image_file) > 1500000000:
				filename = os.path.basename(image_file)
				filename = os.path.splitext(filename)[0]
				print(filename)
				im = Image.open(image_file)
				combined_image, np_perlin = generate_simulated(im, res, alpha, octaves)
				np.random.seed(0)
				plt.imshow(combined_image)
				plt.axis('off')
				plt.savefig("%s.jpg" %filename)
		
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	alpha = 0.5
	octaves = 5
	res = (8,8)
	
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
