import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def get_model(file_name, save_path):
	if not 1 < len(sys.argv) <= 3:
		sys.exit("Incorrect arguments!\n Run: python {} d/l/s (model_name)".format(file_name))

	if len(sys.argv) == 3:
		save_model = sys.argv[2]
		if len(save_model) < 6 or save_model[-5:] != ".ckpt":
			sys.exit("Incorrect file name!")
	else:
		save_model = file_name[:-3]+'.ckpt'

	model_path = os.path.join(save_path, save_model)
	model_exists = os.path.isfile(model_path)

	# Delete file
	if sys.argv[1] == "d":
		if not model_exists:
			sys.exit("[d] File {} doesn't exist!".format(save_model))

		confirm = input("[d] Delete file {}? (y/[n]) ".format(save_model))
		if confirm == "y":
			os.remove(model_path)
			sys.exit("[d] File deleted.")
		else:
			sys.exit("[d] Process canceled.")

	# Save file
	elif sys.argv[1] == "s":
		if model_exists:
			sys.exit("[s] File {} already exists!".format(save_model))

	# Load file
	elif sys.argv[1] == "l":
		if not model_exists:
			sys.exit("[l] File {} doesn't exist!".format(save_model))
		load = True

	# Wrong command
	else:
		sys.exit("Incorrect command. (d: delete, s: save, l: load)")

	return model_exists, save_model

def show_image(imgs, title=None, size=1):
	n = len(imgs)
	if 5 < size < 12:
		size -= 2
	elif 12 <= size:
		size = 10
	fig, ax = plt.subplots(nrows=n, sharex=True, sharey=True, figsize=(1.5*size, 1.5*n))

	if n > 1:
		for i in range(n):
			img = imgs[i] / 2 + 0.5  
			npimg = np.transpose(img.detach().numpy(), (1, 2, 0))
			ax[i].imshow(npimg)
			ax[i].axis('off')
			if title != None:
				ax[i].set_title(title[i])
	else:
		img = imgs[0] / 2 + 0.5  
		npimg = np.transpose(img.detach().numpy(), (1, 2, 0))
		ax.imshow(npimg)
		ax.axis('off')
		if title != None:
			ax.set_title(title[0])


	plt.tight_layout()
	plt.show()