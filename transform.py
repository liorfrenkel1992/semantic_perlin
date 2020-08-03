from PIL import Image
import numpy as np
from perlin2d import generate_perlin_noise_2d, generate_fractal_noise_2d


def generate_simulated(image, alpha):
# This method gets a PIL image, alpha (0-1) and perlin noise as input 
#and returns the combined weighted noisy PIL image
	width, height = image.size
	if width%8 != 0 or height%8 != 0:
		new_image = image.resize((width-(width%8), height-(height%8)))
	
	new_width, new_height = new_image.size
	print('resized image shape: ', new_image.size)
	np_perlin = generate_perlin_noise_2d((new_width,new_height), (8,8))
	np_image = np.asarray(new_image)
	np_perlin = np.repeat(np_perlin[:, :, np.newaxis], 3, axis=2)
	np_image = np.transpose(np_image, (1, 0, 2))
	print('perlin shape: ', np_perlin.shape)
	print('resized image shape: ', np_image.shape)
	np_combined = alpha*np_image + (1-alpha)*np_perlin
	combined = Image.fromarray(np_combined.astype(np.uint8))
	return combined


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	im = Image.open('/data/image_processing/data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg')
	alpha = 0.5
	combined_image = generate_simulated(im, alpha)

	np.random.seed(0)
	plt.imshow(combined_image)
	plt.axis('off')
	plt.savefig('/data/image_processing/data/combined_image.jpg')