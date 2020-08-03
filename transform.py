from PIL import Image
import numpy as np
from perlin2d import generate_perlin_noise_2d, generate_fractal_noise_2d


def generate_simulated(image, res, alpha=0.5, octaves=1):
# This method gets a PIL image, alpha (0-1) and perlin noise as input 
#and returns the combined weighted noisy PIL image
	width, height = image.size
	if width%(res[0]*octaves) != 0 or height%(res[1]*octaves) != 0:
		new_image = image.resize((width-(width%(res[0]*octaves)), height-(height(res[1]*octaves))), 1)
	
	new_width, new_height = new_image.size
	print('resized image shape: ', new_image.size)
	np_perlin = generate_fractal_noise_2d((new_width,new_height), (8,8), octaves)
	np_image = np.asarray(new_image)
	#np_perlin = np.repeat(np_perlin[:, :, np.newaxis], 3, axis=2)
	np_perlin = np.transpose(np_perlin, (1, 0))
	print('perlin shape: ', np_perlin.shape)
	print('resized image shape: ', np_image.shape)
	np_combined = np.zeros(np_image.shape)
	for layer in range(np_image.shape[2]):
		np_combined[:,:,layer] = alpha*np_image[:,:,layer] + (1-alpha)*np_perlin
	combined = Image.fromarray(np_combined.astype(np.uint8))
	return combined, np_perlin


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	im = Image.open('/data/image_processing/data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg')
	alpha = 0.5
	octaves = 2
	res = (8,8)
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

