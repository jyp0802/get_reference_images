import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

from utils import get_model, show_image


def bbox(img):
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]
	return rmin, rmax, cmin, cmax


#### Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
batch_size = 16
test_batch_size = 4
learning_rate = 0.001

data_path = "combined_data"
test_data_path = "test_data"
save_path = "save_model"
load, save_model = get_model(__file__, save_path)


#### Dataset
class RefineImageDataset(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.rgb_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		self.mask_transform = transforms.Compose([
			transforms.ToTensor()
		])
		self.to_pil_transform = transforms.Compose([transforms.ToPILImage()])

		self.adjust_transform = transforms.Compose([
				transforms.Resize((256,256)),
				# transforms.RandomAffine(90, translate=None, scale=None, shear=10, resample=False, fillcolor=0),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

	def __len__(self):
		return sum(os.path.isdir(os.path.join(self.root_dir, str(i))) for i in os.listdir(self.root_dir))

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		# rgb with occlusion, modal patch, eraser, erased patch, predicted amodal patch, predicted rgb, ground truth rgb

		imgs = []
		for i in range(7):
			npy_path = os.path.join(self.root_dir, str(idx), str(i)+".npy")
			imgs.append(np.load(npy_path))

		# for x in imgs:
		# 	img = Image.fromarray(x)
		# 	plt.imshow(img)
		# 	plt.show()
		# 	plt.clf()

		#### ref image from google
		# ref_path = os.path.join(self.root_dir, str(idx), "ref")
		# ref_npy = cv2.imread(ref_path)
		# ref_npy = cv2.cvtColor(ref_npy, cv2.COLOR_BGR2RGB)

		# h, w, _ = ref_npy.shape
		# padding = (int(abs(h-w)/2), 0)
		# if h < w:
		# 	padding = padding[::-1]

		# ref_transform = transforms.Compose([
		# 	transforms.ToPILImage(),
		# 	transforms.Pad(padding, fill=(-1,-1,-1), padding_mode='constant'))
		# ])

		# ref = ref_transform(ref_npy)
		# ref = self.adjust_transform(ref)
		

		#### ref image from adjusting gt image
		gt_ref = imgs[6].copy()

		if np.random.randint(2) == 0:
			gt_ref = 255.0 * (gt_ref / 255.0)**np.random.uniform(0.4, 0.8, 1)[0]
		else:
			gt_ref = 255.0 * (gt_ref / 255.0)**np.random.uniform(1.3, 2.1, 1)[0]
		gt_ref = gt_ref.astype(np.uint8)
		
		if np.random.randint(2) == 0:	
			a,b,c,d = bbox(imgs[4])
			a = np.random.randint(0,a+1)
			b = np.random.randint(b,256)
			c = np.random.randint(0,c+1)
			d = np.random.randint(d,256)
			gt_ref = gt_ref[a:b+1, c:d+1]
			ref = self.to_pil_transform(gt_ref)
			ref = self.adjust_transform(ref)
		else:
			padding = (np.random.randint(4,40),np.random.randint(4,40))
			pad_transform = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Pad(padding, padding_mode='edge')
			])
			ref = pad_transform(gt_ref)
			ref = self.adjust_transform(ref)

		rgb_occ = self.rgb_transform(imgs[0])
		mod_mask = self.mask_transform(imgs[1])
		eraser = self.mask_transform(imgs[2])
		amod_mask = self.mask_transform(imgs[4])
		rgb_pred = self.rgb_transform(imgs[5])
		rgb_gt = self.rgb_transform(imgs[6])
	
		sample = [rgb_occ, mod_mask, eraser, amod_mask, rgb_pred, ref, rgb_gt]

		return sample


trainset = RefineImageDataset(root_dir=data_path)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

testset = RefineImageDataset(root_dir=data_path)
testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=0)


#### Model
class ConvAutoencoderNet(nn.Module):
	def __init__(self):
		super(ConvAutoencoderNet, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(12, 32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 16, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.ConvTranspose2d(16, 3, kernel_size=5, stride=1, padding=2),
			nn.Tanh()
		)

	def forward(self, x):
		x = self.conv(x)
		return x

model = ConvAutoencoderNet().to(device)


#### Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def train():
	total_step = len(trainloader)
	for epoch in range(num_epochs):
		for i, inputs in enumerate(trainloader):
			for idx in range(len(inputs)):
				inputs[idx] = inputs[idx].to(device)

			rgb_occ, mod_mask, eraser, amod_mask, rgb_pred, ref, rgb_gt = inputs
			mod_mask = (mod_mask > 0).float()
			eraser = (eraser > 0).float()
			amod_mask = (amod_mask > 0).float()

			imgs_in = torch.cat([rgb_occ, mod_mask, eraser, amod_mask, rgb_pred, ref], 1)

			#### Masked loss
			rgb_out = model(imgs_in).to(device)
			loss = criterion(rgb_out*eraser, rgb_gt*eraser)
			loss = (loss * eraser).sum()
			mse_loss_val = loss / eraser.sum()

			optimizer.zero_grad()
			mse_loss_val.backward()

			# show_mask = (eraser).cpu().data.numpy()
			# show_mask = np.transpose(show_mask, (0,2,3,1))
			# show_mask = (show_mask * 255).astype(np.uint8)
			# img = Image.fromarray(np.squeeze(show_mask[0]))
			# plt.imshow(img)
			# plt.show()
			# plt.clf()
			
			# show_rgb = (rgb_gt*eraser).cpu().data.numpy()
			# show_rgb = np.transpose(show_rgb, (0,2,3,1))
			# show_rgb = (show_rgb * 255).astype(np.uint8)
			# img = Image.fromarray(show_rgb[0])
			# plt.imshow(img)
			# plt.show()
			# plt.clf()

			
			#### Whole image loss
			# loss.backward()
			# optimizer.step()
			
			if (epoch+1) % 1 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					   .format(epoch+1, num_epochs, i+1, total_step, mse_loss_val.item()))

		if (epoch+1) % 10 == 0:
			torch.save(model.state_dict(), os.path.join(save_path, save_model[:-5]+'_'+str(epoch+1)+save_model[-5:]))



def test():
	model.eval()
	with torch.no_grad():
		for i, inputs in enumerate(testloader):
			if i >= 4:
				break

			for idx in range(len(inputs)):
				inputs[idx] = inputs[idx].to(device)

			rgb_occ, mod_mask, eraser, amod_mask, rgb_pred, ref, rgb_gt = inputs

			imgs_in = torch.cat(inputs[:-1], 1)
			
			# Forward pass
			rgb_out = model(imgs_in).to(device)
			loss = criterion(rgb_out, rgb_gt)
			print(loss.item())

			show_image([torchvision.utils.make_grid(rgb_occ.cpu(), nrow=test_batch_size),
						torchvision.utils.make_grid(rgb_pred.cpu(), nrow=test_batch_size),
						torchvision.utils.make_grid(rgb_out.cpu(), nrow=test_batch_size),
						torchvision.utils.make_grid(rgb_gt.cpu(), nrow=test_batch_size),
						torchvision.utils.make_grid(ref.cpu(), nrow=test_batch_size)],
						["rgb_occ", "rgb_pred", "rgb_out", "rgb_gt", "ref"], size=test_batch_size)


def main():
	torch.multiprocessing.freeze_support()
	if not load:
		train()
		torch.save(model.state_dict(), os.path.join(save_path, save_model))
	if load:
		model.load_state_dict(torch.load(os.path.join(save_path, save_model)))
		test()


if __name__ == '__main__':
	main()