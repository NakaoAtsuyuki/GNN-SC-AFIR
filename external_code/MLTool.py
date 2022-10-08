import numpy as np
import torch
from torch_geometric.data import Data
import Data as MyData
from scipy.spatial import distance
import EQ
import Tool

#transform AFIR result to ML input
dim = 64
mu = np.asarray([i * 0.1 for i in range(dim)])[:, np.newaxis, np.newaxis]
ro = {"+":1, "-":-1}
def eq_to_data (eq, task, G = None, get_y = True):
	idx = eq[0]
	atoms = Tool.ztop(eq[1])
	cord = Tool.ztop(eq[2])
	E = eq[3]
	graph = Tool.ztop(eq[5])
	info = Tool.ztop(task[1])
	result = Tool.ztop(task[2])

	#make 3D graph
	x = torch.tensor([EQ.a_num[a] for a in atoms])
	edge_index = torch.tensor([[[i, j] for i in range(len(atoms))] for j in range(len(atoms))])
	d = distance.cdist(cord, cord, metric='euclidean')
	d = d - np.tile(mu, [1, len(atoms), len(atoms)])
	d = np.exp(-d**2)
	edge_attr = torch.tensor(d).float()

	edge_index = edge_index.reshape(-1, 2).T
	edge_attr = edge_attr.reshape(-1, dim)
	data1 = Data(x = x, edge_index = edge_index, edge_attr=edge_attr)
	
	#add intervention information
	data1.target1_index = torch.tensor(info[0])
	data1.target2_index = torch.tensor(info[1])
	data1.direction = torch.tensor(ro[info[2]]) * torch.ones([len(atoms), 1])
	data1.direction2 = torch.tensor(ro[info[2]]) * torch.ones([1, 1])

	#make 2D graph
	x2 = torch.tensor([EQ.a_num[a] for a in atoms])
	edge_index2 = torch.tensor([[e[0] for e in graph.edges()] + [e[1] for e in graph.edges()], [e[1] for e in graph.edges()] + [e[0] for e in graph.edges()]])
	data2 = Data(x = x2, edge_index = edge_index2)

	#classify success or failure
	if get_y:
		path_EQ = [idx] + [r[2] for r in result if r[0] == "EQ"]
		path_group = [MyData.Data.get_group(G, r) for r in path_EQ]

		path_graph_change = [not path_group[0] == p for p in path_group]
		is_graph_change = any(path_graph_change)
		index_graph_change = path_graph_change.index(True) if is_graph_change else None

		path_EQ_energy = [E] + [r[1] for r in result if r[0] == "EQ"]
		path_TS_energy = [r[1] for r in result if r[0] == "TS"]
		graph_change_TS_energy = path_TS_energy[index_graph_change-1] if is_graph_change else np.inf
		
		return data1, data2, is_graph_change

	else:
		return data1, data2
