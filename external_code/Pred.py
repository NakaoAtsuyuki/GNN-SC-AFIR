#!/home/common_data/buchi/anaconda3/bin/python
##!/home/pyenv/shims/python
import glob
import os
import time
import sqlite3
import traceback

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

import numpy as np

import MLTool
import Tool
import MLModel

ro = {"+":1, "-":-1}

#get jobname
comfile = glob.glob("*.com")
comfile = [comfile[i] for i in np.argsort([len(s) for s in comfile])]
jobname = comfile[0].split(".")[0]

#pytorch dataset for GNN/SC-AFIR
class TaskDataset(Dataset):
	def __init__(self, eq):
		super().__init__()
		self.eq = eq

	def __len__(self):
		return len(self.eq)

	def __getitem__(self, idx):
		return self.eq[idx]

while 1:
	try:
		if os.path.exists(jobname + ".db"):
			conn = sqlite3.connect(jobname + ".db")
		else:
			time.sleep(1)
			continue
		cur = conn.cursor()	

		#wait for EQ selection
		if os.path.exists(jobname + "_SSE.rrm"):
			f = open(jobname + "_SSE.rrm")
			for line in f:
				target_EQ = (int(line[:-1]))
				break
			f.close()
		else:
			conn.close()
			time.sleep(0.01)
			continue

		#read calculated tasks
		started = []
		if os.path.exists(jobname + "_thash.log"):
			f = open(jobname + "_thash.log")
			for line in f:
				tmp = line[:-1]
				started.append(tmp)

		#constructing datalooder
		cur.execute("select * from EQs where idx = ?", (target_EQ,))
		eq = cur.fetchall()[0]

		cur.execute("select * from EQ" + str(target_EQ))
		tasks = cur.fetchall()
		if len(tasks) == 0:
			raise Exception

		thashs = [t[4] for t in tasks]
		if all([t in started for t in thashs]):
			raise Exception

		vals = Tool.pred_base(jobname, target_EQ)
		infos = [[t[0], Tool.ztop(t[1]), t[4]] for t in tasks]
		hvals = [1000 if t in started or v < 0 else 0 for t, v in zip(thashs, vals)]

		idx = []
		x1 = []
		x2 = []
		xx1, xx2 = MLTool.eq_to_data(eq, tasks[0], get_y = False)
		atoms = Tool.ztop(eq[1])
		for i, t in enumerate(tasks):
			if hvals[i] == 1000:
				continue
			else:
				idx.append(i)

			info = Tool.ztop(t[1])

			xx1new = xx1.clone().detach()
			xx1new.target1_index = torch.tensor(info[0])
			xx1new.target2_index = torch.tensor(info[1])
			xx1new.direction = torch.tensor(ro[info[2]]) * torch.ones([len(atoms), 1])
			xx1new.direction2 = torch.tensor(ro[info[2]]) * torch.ones([1, 1])

			x2.append(xx2.clone().detach())
			x1.append(xx1new)

		val_data = TaskDataset([[xx1, xx2] for xx1, xx2 in zip(x1,x2)])
		val_loader = DataLoader(val_data, batch_size = 64, shuffle=False)

		if torch.cuda.is_available():
			device = "cuda"
		else:
			device = "cpu"

		#prediction
		ys = []
		if os.path.exists(jobname + ".pth"):
			model = MLModel.AllGNNrg(device)
			model.load_state_dict(torch.load(jobname + ".pth"))
			model.to(device)
			model.eval()

			f = open(jobname + "_SSE.rrm", "w")
			f.close()
			out_calc = False
			for xval in val_loader:
				x1 = xval[0]
				x2 = xval[1]
				if not out_calc:
					out1 = model.pred_out1(x1)
					out2 = model.pred_out2(x2)
					out_calc = True

				scale = x1.x.shape[0]
				y = model.pred_fast(x1, x2, out1[:scale, :], out2[:scale, :])

				y = y.view(-1)
				ys.extend(list(y.cpu().detach().numpy()))

			for i in range(len(ys)): 
				vals[idx[i]] = ys[i]*0.1**hvals[idx[i]]
		else:
			vals = [np.random.rand()*0.1**h for h in hvals]

		os.remove(jobname + "_SSE.rrm")

		f = open(jobname + "_prediction.rrm", "w")
		for v, info, th, hv in zip(vals, infos, thashs, hvals):
			f.write(str(info[1]) + ":" + str(v) + ":" + th + ":" + str(hv) + "\n")
		f.write("END")
		f.close()
		conn.close()

	except:
		if os.path.exists(jobname + "_SSE.rrm"):
			os.remove(jobname + "_SSE.rrm")

		f = open(jobname + "_prediction.rrm", "w")
		f.write("END")
		f.close()

		traceback.print_exc()
		with open(jobname +"_Ext_ERROR.log", "a") as f:
			traceback.print_exc(file = f)
			f.write("-Prediction-----------------------------\n")

		conn.close()
		continue
