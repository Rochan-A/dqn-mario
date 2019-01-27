# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
	def __init__(self, in_channels=4, num_actions=18):
		super(DQN, self).__init__()

#		Pre-existing code for NN
#		Input is (84, 84, 4)
		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
#		Input is (20, 20, 32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#		Input is (9, 9, 64)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#		Input is (7, 7, 64)
		self.fc4 = nn.Linear(7 * 7 * 64, 512)
		self.fc5 = nn.Linear(512, num_actions)

#		Code from A3C
#		self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
#		self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#		self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#		self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#		self.fc4 = nn.Linear(32 * 6 * 6, 512)
#		self.fc5 = nn.Linear(512, num_actions)

	def forward(self, inputs):

#		Pre-existing code for NN
		x = F.relu(self.conv1(inputs))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.fc4(x.view(x.size(0), -1)))

#		Code from A3C
#		x = F.elu(self.conv1(inputs))
#		x = F.elu(self.conv2(x))
#		x = F.elu(self.conv3(x))
#		x = F.elu(self.conv4(x))
#		x = F.relu(self.fc4(x.view(x.size(0), -1)))

		return self.fc5(x)