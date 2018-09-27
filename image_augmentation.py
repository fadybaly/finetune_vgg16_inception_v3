import os
from glob import glob
from datetime import datetime
from shutil import copyfile
from imgaug import augmenters as iaa
from scipy.misc import imsave, imread

INPUT = '/home/fady/Desktop/test/'
OUTPUT = ''
WHITE_LIST_FORMAT = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'JPG')
ITERATIONS = 10


def check_dir_or_create(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)


# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

augmenters = [
	iaa.Fliplr(0.5),  # horizontal flips
	iaa.Crop(percent=(0, 0.1)),  # random crops
	# Strengthen or weaken the contrast in each image.
	iaa.ContrastNormalization((0.75, 1.5)),
	# Make some images brighter and some darker.
	# In 20% of all cases, we sample the multiplier once per channel,
	# which can end up changing the color of the images.
	iaa.Multiply((0.8, 1.2), per_channel=0.2),
	# Apply affine transformations to each image.
	# Scale/zoom them, translate/move them, rotate them and shear them.
	iaa.Affine(
		scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
		translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
		rotate=(-25, 25),
		shear=(-8, 8)
	)
]

seq = iaa.Sequential(augmenters, random_order=True)

files = [file for directory in os.walk(INPUT)
				for file in glob(os.path.join(directory[0], '*')) if os.path.isfile(file)]

files = [f for f in files if f.endswith(WHITE_LIST_FORMAT)]
classes = [os.path.basename(os.path.dirname(x)) for x in files]
classes_set = set(classes)
for _class in classes_set:
	_dir = os.path.join(OUTPUT, _class)
	check_dir_or_create(_dir)

batches = []
BATCH_SIZE = 1
batches_count = len(files) // BATCH_SIZE + 1
for i in range(batches_count):
	batches.append(files[BATCH_SIZE * i:BATCH_SIZE * (i + 1)])

images = []
for i in range(ITERATIONS):
	print(i, datetime.time(datetime.now()))
	for batch in batches:
		images = []
		for file in batch:
			img = imread(file)
			images.append(img)
		images_aug = seq.augment_images(images)
		for file, image_aug in zip(batch, images_aug):
			root, ext = os.path.splitext(file)
			new_filename = root + '_{}'.format(i) + ext
			new_path = new_filename.replace(INPUT, OUTPUT, 1)

			imsave(new_path, image_aug)

for file in files:
	dst = file.replace(INPUT, OUTPUT)
	copyfile(file, dst)
