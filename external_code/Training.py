#!/home/common_data/buchi/anaconda3/bin/python
##!/home/pyenv/shims/python
import glob
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import sqlite3
import MLModel
import MLTool
import Tool
import time
import traceback

#get jobname
comfile = glob.glob("*.com")
comfile = [comfile[i] for i in np.argsort([len(s) for s in comfile])]
jobname = comfile[0].split(".")[0]

import Data as MyData
import EQ
import sys
import pickle

dim = 64
mu = np.asarray([i * 0.1 for i in range(dim)])[:, np.newaxis, np.newaxis]
ro = {"+":1, "-":-1}

from scipy.spatial import distance

#pytorch dataset for GNN/SC-AFIR
class TaskDataset(Dataset):
	def __init__(self, eq, y):
		super().__init__()
		self.eq = eq
		self.y = y
		
	def __len__(self):
		return len(self.eq)

	def __getitem__(self, idx):
		return self.eq[idx], torch.tensor(self.y[idx])

get_time = 0
x1 = []
x2 = []
y = []

#model construction
if torch.cuda.is_available():
	device = "cuda"
else:
	device = "cpu"

model = MLModel.AllGNNrg(device)

#load model for transfer learning
tf_flag = False
if tf_flag:
	model.load_state_dict(torch.load(jobname + ".pth"))
	for param in model.parameters():
		if param.shape[0] == 8:
			break
		param.requires_grad = False

model.to(device)
optimizer = optim.RAdam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()
while 1:
	try:
		#updata dataloader
		log = open(jobname + "_ML.log", "a")
		conn = sqlite3.connect(jobname + ".db")
		cur = conn.cursor()

		cur.execute("select count(*) from EQs")
		num = cur.fetchall()[0][0]
		if num == 0:
			continue
			conn.close()

		cur.execute("select graph_group_member from Main")
		G = Tool.ztop(cur.fetchall()[0][0])
		tmp_time = time.time()
		new_data = []
		for i in range(num):
			cur.execute("select * from EQ" + str(i) + " where state = 2 and time > ? and time <= ?", (get_time, tmp_time))
			tasks = cur.fetchall()
			new_data.extend([(i, t) for t in tasks])

		for data in new_data:
			cur.execute("select * from EQs where idx = ?", (data[0],))
			eq = cur.fetchall()[0]
			
			xx1, xx2, yy = MLTool.eq_to_data(eq, data[1], G)

			x1.append(xx1)
			x2.append(xx2)
			y.append(yy)

		if len(x1) == 0:
			continue
			conn.close()

		get_time = np.max([t[1][5] for t in new_data]) if len(new_data) > 0 else get_time
		conn.close()

		train_data = TaskDataset([[xx1, xx2] for xx1, xx2 in zip(x1, x2)], [yy for yy in y])
		train_loader = DataLoader(train_data, batch_size = 64, shuffle=True)

		#1 epoch model training
		model.train()

		log = open(jobname + "_ML.log", "a")
		log.write("----------------------------------------\n")
		log.write("data : " + str(len(x1)) + "\n")
		log.close()

		loss_sum = 0
		c = 0
		for xtrain, ytrain in train_loader:
			xtrain[0].to(device)
			xtrain[1].to(device)
			ytrain.to(device)

			optimizer.zero_grad()
			pred = model(xtrain[0], xtrain[1]).view([-1])
			loss = loss_fn(pred, ytrain.float())
			loss.backward()
			optimizer.step()
			loss_sum += loss.item()
			c += 1

		log = open(jobname + "_ML.log", "a")
		log.write(str(i) + " : " + str(loss_sum/c) + "\n")
		log.close()
		
		torch.save(model.to('cpu').state_dict(), jobname + ".pth" )
	
	except:
		traceback.print_exc()
		with open(jobname +"_Ext_ERROR.log", "a") as f:
			traceback.print_exc(file = f)
			f.write("-training-----------------------------\n")

		continue
