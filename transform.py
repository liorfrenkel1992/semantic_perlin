from PIL import Image
import numpy as np
import perlin2d

# This method gets a PIL image, alpha (0-1) and perlin noise as input 
#and returns the combined weighted noisy PIL image
def generate_simulated(image, alpha):
	width, height = image.size
	if width%8 != 0 or height%8 != 0:
		new_image = image.resize((width-(width%8), height-(height%8)))
	#print('width: ', w)
	#print('height:', h)
	new_width, new_height = new_image.size
	np_perlin = generate_fractal_noise_2d((new_width,new_height), (8,8), 5)
	np_image = np.array(new_image)
	np_combined = alpha*np_image + (1-alpha)*np_perlin
	combined = Image.fromarray(np_combined)
	return combined


im = Image.open('/data/image_processing/data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg')
alpha = 0.5
combined_image = generate_simulated(im, alpha)

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	np.random.seed(0)
	plt.imshow(noise, cmap='gray', interpolation='lanczos')
	plt.axis('off')
	plt.savefig('/data/image_processing/data/combined_image.jpg')
