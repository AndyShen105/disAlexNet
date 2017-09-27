import os
import random
import tensorflow as tf 
import numpy as np
from scipy.io import loadmat
from PIL import Image


################ TensorFlow standard operations wrappers #####################
def weight(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.01)
	w = tf.Variable(initial, name=name)
	tf.add_to_collection('weights', w)
	return w

def bias(value, shape, name):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, stride, padding):
	return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)

def max_pool2d(x, kernel, stride, padding):
	return tf.nn.max_pool(x, ksize=kernel, strides=stride, padding=padding)

def lrn(x, depth_radius, bias, alpha, beta):
	return tf.nn.local_response_normalization(x, depth_radius, bias, alpha, beta)

def relu(x):
	return tf.nn.relu(x)

def batch_norm(x):
	epsilon = 1e-3
	batch_mean, batch_var = tf.nn.moments(x, [0])
	return tf.nn.batch_normalization(x, batch_mean, batch_var, None, None, epsilon)

################ batch creation functions #####################

def onehot(index):
	""" It creates a one-hot vector with a 1.0 in
		position represented by index 
	"""
	onehot = np.zeros(10)
	onehot[index] = 1.0
	return onehot

def read_batch(batch_size, images_source, wnid_labels):
	""" It returns a batch of single images (no data-augmentation)

		ILSVRC 2012 training set folder should be srtuctured like this: 
		ILSVRC2012_img_train
			|_n01440764
			|_n01443537
			|_n01484850
			|_n01491361
			|_ ... 

		Args:
			batch_size: need explanation? :)
			images_sources: path to ILSVRC 2012 training set folder
			wnid_labels: list of ImageNet wnid lexicographically ordered

		Returns:
			batch_images: a tensor (numpy array of images) of shape [batch_size, width, height, channels] 
			batch_labels: a tensor (numpy array of onehot vectors) of shape [batch_size, 1000]
	"""
	batch_images = []
	batch_labels = []

	for i in range(batch_size):
		# random class choice 
		# (randomly choose a folder of image of the same class from a list of previously sorted wnids)
		class_index = random.randint(0, 9)

		folder = wnid_labels[class_index]
		batch_images.append(read_image(os.path.join(images_source, folder)))
		batch_labels.append(onehot(class_index))

	np.vstack(batch_images)
	np.vstack(batch_labels)
	return batch_images, batch_labels

def read_image(images_folder):
	""" It reads a single image file into a numpy array and preprocess it

		Args:
			images_folder: path where to random choose an image

		Returns:
			im_array: the numpy array of the image [width, height, channels]
	"""
	# random image choice inside the folder 
	# (randomly choose an image inside the folder)
	image_path = os.path.join(images_folder, random.choice(os.listdir(images_folder)))
	
	# load and normalize image
	im_array = preprocess_image(image_path)
	#im_array = read_k_patches(image_path, 1)[0]
		
	return im_array

def preprocess_image(image_path):
	""" It reads an image, it resize it to have the lowest dimesnion of 256px,
		it randomly choose a 224x224 crop inside the resized image and normilize the numpy 
		array subtracting the ImageNet training set mean

		Args:
			images_path: path of the image

		Returns:
			cropped_im_array: the numpy array of the image normalized [width, height, channels]
	"""
	IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format

	img = Image.open(image_path).convert('RGB')

	# resize of the image (setting lowest dimension to 256px)
	if img.size[0] < img.size[1]:
		h = int(float(256 * img.size[1]) / img.size[0])
		img = img.resize((256, h), Image.ANTIALIAS)
	else:
		w = int(float(256 * img.size[0]) / img.size[1])
		img = img.resize((w, 256), Image.ANTIALIAS)

	# random 244x224 patch
	x = random.randint(0, img.size[0] - 224)
	y = random.randint(0, img.size[1] - 224)
	img_cropped = img.crop((x, y, x + 224, y + 224))

	cropped_im_array = np.array(img_cropped, dtype=np.float32)

	for i in range(3):
		cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]
	return cropped_im_array

def read_k_patches(image_path, k):
	""" It reads k random crops from an image

		Args:
			images_path: path of the image
			k: number of random crops to take

		Returns:
			patches: a tensor (numpy array of images) of shape [k, 224, 224, 3]

	"""
	IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format

	img = Image.open(image_path).convert('RGB')

	# resize of the image (setting largest border to 256px)
	if img.size[0] < img.size[1]:
		h = int(float(256 * img.size[1]) / img.size[0])
		img = img.resize((256, h), Image.ANTIALIAS)
	else:
		w = int(float(256 * img.size[0]) / img.size[1])
		img = img.resize((w, 256), Image.ANTIALIAS)

	patches = []
	for i in range(k):
		# random 244x224 patch
		x = random.randint(0, img.size[0] - 224)
		y = random.randint(0, img.size[1] - 224)
		img_cropped = img.crop((x, y, x + 224, y + 224))

		cropped_im_array = np.array(img_cropped, dtype=np.float32)

		for i in range(3):
			cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]

		patches.append(cropped_im_array)

	np.vstack(patches)
	return patches

""" reading a batch of validation images from the validation set, 
	groundthruths label are inside an annotations file """
def read_validation_batch(batch_size, validation_source, annotations):
	batch_images_val = []
	batch_labels_val = []

	images_val = sorted(os.listdir(validation_source))

	# reading groundthruths labels
	with open(annotations) as f:
		gt_idxs = f.readlines()
		gt_idxs = [(int(x.strip()) - 1) for x in gt_idxs]

	for i in range(batch_size):
		# random image choice
		idx = random.randint(0, len(images_val) - 1)

		image = images_val[idx]
		batch_images_val.append(preprocess_image(os.path.join(validation_source, image)))
		batch_labels_val.append(onehot(gt_idxs[idx]))

	np.vstack(batch_images_val)
	np.vstack(batch_labels_val)
	return batch_images_val, batch_labels_val

def read_validation_batch_V2(n_val, validation_source, annotations):
	batch_images_val = []
	batch_labels_val = []


	# reading groundthruths labels
	f=open(annotations)
	for item in f.readlines():
	    pic = item.split(",")[1].strip()
	    while len(pic) < 5:
		pic ="0"+ pic
	    idx = item.split(",")[2]
	    batch_images_val.append(preprocess_image("/root/data/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_000"+pic+".JPEG"))
	    batch_labels_val.append(onehot(int(idx)-1))
	    np.vstack(batch_images_val)
 	    np.vstack(batch_labels_val)
	return batch_images_val, batch_labels_val

################ Other helper procedures #####################

def load_imagenet_meta(meta_path):
	""" It reads ImageNet metadata from ILSVRC 2012 dev tool file

		Args:
			meta_path: path to ImageNet metadata file

		Returns:
			wnids: list of ImageNet wnids labels (as strings)
			words: list of words (as strings) referring to wnids labels and describing the classes 

	"""
	meta_file = open(meta_path, "r")
	meta_data = meta_file.readlines()
	wnids = np.squeeze(np.array([s.strip() for s in meta_data]))
	#words = np.squeeze(np.array([s.split(",")[1] for s in meta_data]))
	return wnids

def read_test_labels(annotations_path):
	""" It reads groundthruth labels from ILSRVC 2012 annotations file

		Args:
			annotations_path: path to the annotations file

		Returns:
			gt_labels: a numpy vector of onehot labels
	"""
	gt_labels = []

	# reading groundthruths labels from ilsvrc12 annotations file
	with open(annotations_path) as f:
		gt_idxs = f.readlines()
		gt_idxs = [(int(x.strip()) - 1) for x in gt_idxs]

	for gt in gt_idxs:
		gt_labels.append(onehot(gt))

	np.vstack(gt_labels)

	return gt_labels

def format_time(time):
	""" It formats a datetime to print it

		Args:
			time: datetime

		Returns:
			a formatted string representing time
	"""
	m, s = divmod(time, 60)
	h, m = divmod(m, 60)
	d, h = divmod(h, 24)
	return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))

def imagenet_size(im_source):
	""" It calculates the number of examples in ImageNet training-set

		Args:
			im_source: path to ILSVRC 2012 training set folder

		Returns:
			n: the number of training examples

	"""
	n = 0
	for d in os.listdir(im_source):
		for f in os.listdir(os.path.join(im_source, d)):
			n += 1
	return n
