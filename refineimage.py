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

from utils import get_model, show_image


#### Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 500
batch_size = 16
test_batch_size = 4
learning_rate = 0.001

data_path = "new_data"
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

	def __len__(self):
		return sum(os.path.isdir(os.path.join(self.root_dir, str(i))) for i in os.listdir(self.root_dir))

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		# rgb with occlusion
		# modal patch
		# eraser
		# erased patch
		# predicted amodal patch
		# predicted rgb
		# ground truth rgb

		imgs = []
		for i in range(7):
			npy_path = os.path.join(self.root_dir, str(idx), str(i)+".npy")
			imgs.append(np.load(npy_path))

		ref_path = os.path.join(self.root_dir, str(idx), "ref")
		ref_npy = cv2.imread(ref_path)
		ref_npy = cv2.cvtColor(ref_npy, cv2.COLOR_BGR2RGB)

		h, w, _ = ref_npy.shape
		padding = (int(abs(h-w)/2), 0)
		if h < w:
			padding = padding[::-1]

		ref_transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Pad(padding, fill=(-1,-1,-1), padding_mode='constant'),
			transforms.Resize(imgs[0].shape[:-1]),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		rgb_occ = self.rgb_transform(imgs[0])
		mod_mask = self.mask_transform(imgs[1])
		amod_mask = self.mask_transform(imgs[4])
		rgb_pred = self.rgb_transform(imgs[5])
		rgb_gt = self.rgb_transform(imgs[6])
		ref = ref_transform(ref_npy)

		sample = [rgb_occ, mod_mask, amod_mask, rgb_pred, ref, rgb_gt]

		return sample


trainset = RefineImageDataset(root_dir=data_path)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = RefineImageDataset(root_dir=data_path)
testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=0)


#### Model
class ConvAutoencoderNet(nn.Module):
	def __init__(self):
		super(ConvAutoencoderNet, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(11, 32, kernel_size=5, stride=1, padding=2),
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

			rgb_occ, mod_mask, amod_mask, rgb_pred, ref, rgb_gt = inputs

			imgs_in = torch.cat(inputs[:-1], 1)
			
			# Forward pass
			rgb_out = model(imgs_in).to(device)
			loss = criterion(rgb_out, rgb_gt)
			
			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			if (epoch+1) % 1 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


def test():
	model.eval()
	with torch.no_grad():
		for i, inputs in enumerate(testloader):
			if i >= 4:
				break

			for idx in range(len(inputs)):
				inputs[idx] = inputs[idx].to(device)

			rgb_occ, mod_mask, amod_mask, rgb_pred, ref, rgb_gt = inputs

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