import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GINEConv, GINConv
from torch_geometric.utils import softmax
from torch_geometric.nn.glob import global_add_pool
import math
import time

dim = 64

class MLP(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.nn1 = torch.nn.Linear(dim , dim)
		self.nn2 = torch.nn.Linear(dim , dim)
		
	def forward(self, x):
		x = F.relu(self.nn1(x))
		return self.nn2(x)

#The class of machine learning model
class AllGNNrg(torch.nn.Module):
	head = 8
	num = 4
	
	def __init__(self, device):
		super().__init__()

		#atom embedding		
		self.atom_emb1 = torch.nn.Embedding(100, dim)
		self.atom_emb2 = torch.nn.Embedding(100, dim)
		
		#GIN layer
		self.mlps1 = []
		self.gnns1 = []
		self.bns1 = []
		
		self.mlps2 = []
		self.gnns2 = []
		self.bns2 = []
		
		for i in range(self.num):
			self.mlps1.append(MLP())
			self.gnns1.append(GINEConv(nn=self.mlps1[-1]))
			self.bns1.append(torch.nn.BatchNorm1d(dim))
			
			self.mlps2.append(MLP())
			self.gnns2.append(GINConv(nn=self.mlps2[-1]))
			self.bns2.append(torch.nn.BatchNorm1d(dim))
			
		self.mlps1 = torch.nn.ModuleList(self.mlps1)
		self.gnns1 = torch.nn.ModuleList(self.gnns1)
		self.bns1 = torch.nn.ModuleList(self.bns1)
		
		self.mlps2 = torch.nn.ModuleList(self.mlps2)
		self.gnns2 = torch.nn.ModuleList(self.gnns2)
		self.bns2 = torch.nn.ModuleList(self.bns2)
		

		#multi-head attention layer
		self.Wqs1 = []
		self.Wks1 = []
		self.Wvs1 = []
		
		self.Wqs2 = []
		self.Wks2 = []
		self.Wvs2 = []
		
		for i in range(self.head):
			self.Wqs1.append(torch.nn.Linear(dim+1 , int(dim/self.head), bias = False))
			self.Wks1.append(torch.nn.Linear(dim+1 , int(dim/self.head), bias = False))
			self.Wvs1.append(torch.nn.Linear(dim+1 , int(dim/self.head), bias = False))
			
			self.Wqs2.append(torch.nn.Linear(dim+1 , int(dim/self.head), bias = False))
			self.Wks2.append(torch.nn.Linear(dim+1 , int(dim/self.head), bias = False))
			self.Wvs2.append(torch.nn.Linear(dim+1 , int(dim/self.head), bias = False))
			
		self.Wqs1 = torch.nn.ModuleList(self.Wqs1)
		self.Wks1 = torch.nn.ModuleList(self.Wks1)
		self.Wvs1 = torch.nn.ModuleList(self.Wvs1)
		
		self.Wqs2 = torch.nn.ModuleList(self.Wqs2)
		self.Wks2 = torch.nn.ModuleList(self.Wks2)
		self.Wvs2 = torch.nn.ModuleList(self.Wvs2)
		
		#full connected layer
		self.fc11 = torch.nn.Linear(dim + 1 , int(dim/2))
		self.gelu1 = torch.nn.GELU()
		self.DO1 = torch.nn.Dropout(0.5)
		self.fc12 = torch.nn.Linear(int(dim/2) , 1)
		
		self.device = device

	def forward(self, x1, x2):
		out1 = self.pred_out1(x1)
		out2 = self.pred_out2(x2)
		out = self.pred_fast(x1, x2, out1, out2)
		return out

	#calculate GIN for 3D graph
	def pred_out1(self, x1):
		out1 = self.atom_emb1(x1.x)
		for i in range(self.num):
			out1_tmp = self.gnns1[i](x = out1, edge_index = x1.edge_index, edge_attr = x1.edge_attr)
			out1_tmp = self.bns1[i](out1_tmp)
			out1 = out1 + out1_tmp

		return out1

	#calculate GIN for 2D graph
	def pred_out2(self, x2):
		out2 = self.atom_emb2(x2.x)
		for i in range(self.num):
			out2_tmp = self.gnns2[i](x = out2, edge_index = x2.edge_index)
			out2_tmp = self.bns2[i](out2_tmp)
			out2 = out2 + out2_tmp

		return out2
	
	def pred_fast(self, x1, x2, out1, out2):
		out1 = torch.cat([out1, x1.direction], dim = 1)
		out2 = torch.cat([out2, x1.direction], dim = 1)

		outs1 = []
		for i in range(self.head):
			Q = self.Wqs1[i](out1)
			V = self.Wvs1[i](out1)

			K1 = self.Wks1[i](out1[x1.target1_index])
			QK1 = torch.sum(torch.mm(Q, K1.transpose(-2, -1)) * F.one_hot(x1.batch), axis = 1)  / math.sqrt(int(dim/self.head))
			sQK1 = softmax(QK1, x1.batch)

			K2 = self.Wks1[i](out1[x1.target2_index])
			QK2 = torch.sum(torch.mm(Q, K2.transpose(-2, -1)) * F.one_hot(x1.batch), axis = 1)  / math.sqrt(int(dim/self.head))
			sQK2 = softmax(QK2, x1.batch)
	
			tmp = (sQK1 + sQK2).view([-1,1]) * V
			outs1.append(global_add_pool(tmp, x1.batch))

		outs2 = []
		for i in range(self.head):
			Q = self.Wqs2[i](out2)
			V = self.Wvs2[i](out2)
		
			K1 = self.Wks2[i](out2[x1.target1_index])
			QK1 = torch.sum(torch.mm(Q, K1.transpose(-2, -1)) * F.one_hot(x1.batch), axis = 1)  / math.sqrt(int(dim/self.head))
			sQK1 = softmax(QK1, x2.batch)

			K2 = self.Wks2[i](out2[x1.target2_index])
			QK2 = torch.sum(torch.mm(Q, K2.transpose(-2, -1)) * F.one_hot(x2.batch), axis = 1)  / math.sqrt(int(dim/self.head))
			sQK2 = softmax(QK2, x2.batch)

			tmp = (sQK1 + sQK2).view([-1,1]) * V
			outs2.append(global_add_pool(tmp, x2.batch))

		out1 = torch.cat(outs1, axis = 1)
		out2 = torch.cat(outs2, axis = 1)
		out = torch.cat([out1+out2, x1.direction2], dim = 1)
		return torch.sigmoid(self.fc12(self.DO1(self.gelu1(self.fc11(out)))))
