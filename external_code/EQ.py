import os
import time
import re
import sqlite3
import pybel
import glob

import networkx as nx
import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem

import Task
import Tool

#covalent radius
r_cov = {"H" :0.32, "He":0.46, "Li":1.33e-3, "Be":1.02e-3, "B" :0.85, "C" :0.75, "N" :0.71, "O" :0.63, "F" :0.64, "Ne":0.67,
         "Na":1.55e-3, "Mg":1.39e-3, "Al":1.26e-3, "Si":1.16, "P" :1.11, "S" :1.03, "Cl":0.99, "Ar":0.67, "K" :1.96e-3, "Ca":1.71e-3,
         "Sc":1.48e-3, "Ti":1.36e-3, "V" :1.34e-3, "Cr":1.22e-3, "Mn":1.19e-3, "Fe":1.16e-3, "Co":1.11e-3, "Ni":1.10e-3, "Cu":1.12e-3, "Zn":1.18e-3,
         "Ga":1.24, "Ge":1.24, "As":1.21, "Se":1.16, "Br":1.14, "Kr":1.17, "Rb":2.10, "Sr":1.85, "Y" :1.63, "Zr":1.54,
         "Nb":1.47, "Mo":1.38, "Tc":1.28, "Ru":1.25, "Rh":1.25, "Pd":1.20, "Ag":1.28, "Cd":1.36, "In":1.42, "Sn":1.40,
         "Sb":1.40, "Te":1.36, "I" :1.33, "Xe":1.31, "Cs":2.32, "Ba":1.96, "La":1.80, "Ce":1.63, "Pr":1.76, "Nd":1.74,
         "Pm":1.73, "Sm":1.72, "Eu":1.68, "Gd":1.69, "Tb":1.68, "Dy":1.67, "Ho":1.66, "Er":1.65, "Tm":1.64, "Yb":1.70,
         "Lu":1.62, "Hf":1.52, "Ta":1.46, "W" :1.37, "Re":1.31, "Os":1.29, "Ir":1.22, "Pt":1.23, "Au":1.24, "Hg":1.33,
         "Tl":1.44, "Pb":1.44, "Bi":1.51, "Po":1.45, "At":1.47, "Rn":1.42, "Fr":2.23, "Ra":2.01, "Ac":1.86, "Th":1.75,
         "Pa":1.69, "U" :1.70, "Np":1.71, "Pu":1.72, "Am":1.66, "Cm":1.66, "Bk":1.66, "Cf":1.68, "Es":1.65, "Fm":1.67}
#atomic number
a_num = {"H" :  1, "He":  2, "Li":  3, "Be":  4, "B" :  5, "C" :  6, "N" :  7, "O" :  8, "F" :  9, "Ne": 10,
         "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P" : 15, "S" : 16, "Cl": 17, "Ar": 18, "K" : 19, "Ca": 20,
         "Sc": 21, "Ti": 22, "V" : 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
         "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y" : 39, "Zr": 40,
         "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
         "Sb": 51, "Te": 52, "I" : 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
         "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
         "Lu": 71, "Hf": 72, "Ta": 73, "W" : 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
         "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
         "Pa": 91, "U" : 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm":100}

#rdkit stereochemistry tag     
chipat= {"CHI_TETRAHEDRAL_CW":"@", "CHI_TETRAHEDRAL_CCW":"@@", "CHI_UNSPECIFIED":""}
ezpat = {"STEREONONE":"", "STEREOZ":"Z", "STEREOE":"E"}

#The class for constructing EQ information
class EQ:

	#idx: index of EQ
	#atoms: list of atom
	#X: coordinate of atoms
	#E: energy of EQ
	def __init__(self, idx, atoms, X, E = None):
		self.idx = idx
		self.atoms = atoms
		self.X = X
		self.E = E
		
		#create 2D structure
		self.graph, self.smiles = self.make_graph()

	#translate mol object to networkx graph
	@classmethod
	def mol_to_graph(cls, mol):
		G = nx.Graph()
		for i, a in enumerate(mol.GetAtoms()):
			name = a.GetSymbol() + chipat[str(a.GetChiralTag())]
			G.add_node(i, color = name)
        
		for b in mol.GetBonds():
			G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), color = str(b.GetBondType()), dir = b.GetStereo())
		
		return G

	#create 2D structure and SMILES
	def make_graph(self):
		#translate xyz to pybel object
		scr = str(len(self.atoms)) + "\n\n"
		for i in range(len(self.atoms)):
			scr += self.atoms[i] + " " + str(self.X[i, 0]) + " " + str(self.X[i, 1]) + " " + str(self.X[i, 2]) + "\n"
		mol = pybel.readstring("xyz", scr)
		
		#translate pybel object to mol file
		mol.write("mol", "outputfile.mol", overwrite=True)

		#sanitize mol object
		mol = Chem.MolFromMolFile('outputfile.mol', sanitize = False)
		mol.UpdatePropertyCache(strict=False)
		AllChem.AssignStereochemistry(mol)
	
		#create networkx graph and SMILES
		G = EQ.mol_to_graph(mol)
		smiles = Chem.MolToSmiles(mol)
	
		return G, smiles
	
	#read files related to an EQ
	@classmethod
	def reload(cls, jobname, num, numb):
		read_error = False
	
		tmp_time = time.time()
		
		#read jobname_EQ*_iPATH.rrm
		EQ.read_iPATH(num, jobname)
		
		#read jobname_P*_EQ*.log
		read_error = EQ.read_log(num, jobname, numb)
	
		if not read_error:
			conn = sqlite3.connect(jobname + ".db")
			cur = conn.cursor()
			cur.execute("update EQs set reload_time = ? where idx = ?", (tmp_time, num))
			conn.commit()
			conn.close()
		
	#read jobname_EQ*_iPATH.rrm
	@classmethod
	def read_iPATH(cls, num, jobname):
		conn = sqlite3.connect(jobname + ".db")
		cur = conn.cursor()

		cur.execute("select info from EQ" + str(num))
		tmp = cur.fetchall()
		tasks = [Tool.ztop(s[0]) for s in tmp]

		ro = {"plus":"+", "minus":"-", "global":"+"}
		iPATH = jobname + "_EQ" + str(num) + "_tPATH.rrm"
		if os.path.exists(iPATH):
			f = open(iPATH)
			for idx, line in enumerate(f):
				tmp = line.replace( '\n' , '' ).split(" ")
				job = [int(tmp[0]), int(tmp[1]), ro[tmp[2]]]
				
				job_rowid = EQ.find_task(job, tasks)
				cur.execute("update EQ" + str(num) + " set idx = ? where rowid = ?", (idx, job_rowid))
			f.close()
		conn.commit()
		conn.close()

	#read jobname_P*_EQ*.log files
	@classmethod
	def read_log(cls, num, jobname, numb):
		main_log = jobname + "_EQ" + str(num) + ".log"
		conn = sqlite3.connect(jobname + ".db")
		cur = conn.cursor()

		cur.execute("select numb from Main")
		numb = Tool.ztop(cur.fetchall()[0][0])

		state = False
		if os.path.exists(main_log):
			read_error = EQ.read_log_info(cur, num, main_log)
			if read_error:
				state = True
			
		for p in numb.keys():
			log_file = jobname + "_P" + str(p) + "_EQ" + str(get_key_from_value(numb[p],num)) + ".log"
			read_error = EQ.read_log_info(cur, num, log_file, numb, p)
			if read_error:
				state = True

		conn.commit()
		conn.close()

		return state
	
	#read jobname_P*_EQ*.log
	@classmethod
	def read_log_info(cls, cur, num, filename, numb = None, P = None):
		cur.execute("select reload_time from EQs where rowid = ?", (num,))
		reload_time = cur.fetchall()[0][0]
		if not os.path.exists(filename) or len(glob.glob(str(num) + "_*[0-9]")) < 0:
			return
		f = open(filename,"r")
		
		info = {}
		read_error = False
		
		mode = 0 
		read = 0 
		regex_float = re.compile('-*\d+.\d+')
		
		for line in f:
			if "FIND SC-AFIR-PATH" in line:
				check = Task.Task.set_result(info, num, cur)
				if not check:
					read_error = True
				
				info = {}
				info["idx"] = int(line.split(" ")[-1][:-1])
				
				mode = 1
				
			if mode == 1 and "Start AFIR-minimization" in line:
				info["path"] = []
				mode = 2
				
			if mode == 2:
				if "THE NUMBER OF APPROXIMATE" in line:
					mode = 3
					
				if read == 1 and "ENERGY" in line:
					e = float(regex_float.findall(line)[1])
					info["path"].append([typ, EQ(-1, atoms, np.asarray(xs), e)])
					read = 0
					
				if read == 1:
					elements = line.split('\t')
					atoms.append(elements[0].replace(' ', ''))
					xs.append([float(e.replace(' ', '')) for e in elements[1:]])
					
				if "Approximate" in line:
					typ = "TS" if "TS" in line else "EQ"
					atoms = []
					xs = []
					read = 1
			
			if mode == 3 and "Start LUP-optimization" in line:
				mode = 4
				info["path"] = []
				
			if mode == 4:
				if "THE NUMBER OF APPROXIMATE" in line:
					mode = 5
					
				if read == 1 and "ENERGY" in line:
					e = float(regex_float.findall(line)[1])
					info["path"].append([typ, EQ(-1, atoms, np.asarray(xs), e)])
					read = 0
					
				if read == 1:
					elements = line.split('\t')
					atoms.append(elements[0].replace(' ', ''))
					xs.append([float(e.replace(' ', '')) for e in elements[1:]])
					
				if "Approximate" in line:
					typ = "TS" if "TS" in line else "EQ"
					atoms = []
					xs = []
					read = 1
					
			if (mode == 3 or mode == 5) and "Start MIN-optimization" in line:
				EQnum = int(line.split("-")[-1][:-1])
				
				eq_count = 0
				pt_count = 0
				for tmp in info["path"]:
					if tmp[0] == "EQ":
						if EQnum == eq_count:
							EQidx = pt_count
							break
						eq_count += 1
					pt_count += 1
				mode = 6
				
			if mode == 6:
				if "THE NUMBER OF APPROXIMATE" in line:
					mode = 5

				if "Maximum number of iteration was exceeded" in line or "The structure is dissociating" in line or "optimization failed" in line:
					mode = 5
					info["path"][EQidx].append(None)
					
				if read == 1 and "ENERGY" in line:
					e = float(regex_float.findall(line)[1])
					info["path"][EQidx][1] = EQ(-1, atoms, np.asarray(xs), e)
					mode = 7
					read = 0
					
				if read == 1:
					elements = line.split('\t')
					atoms.append(elements[0].replace(' ', ''))
					xs.append([float(e.replace(' ', '')) for e in elements[1:]])
					
				if "Optimized structure" in line:
					atoms = []
					xs = []
					read = 1
					
			if mode == 7 and "was found:" in line:
				regex_int = re.compile('\d+')
				i = int(regex_int.findall(line)[-1])
				
				if not numb == None:
					if i in numb[P].keys():
						i = numb[P][i]
					else:
						mode = 5
						f.close()
						return True
				
				info["path"][EQidx].append(i)
				mode = 5

		res = Task.Task.set_result(info, num, cur)
		if not res:
			read_error = True

		f.close()
		return read_error
		
	#get task index
	@classmethod
	def find_task(cls, job, tasks):
		for i, t in enumerate(tasks):
			if EQ.match_job_info(t, job):
				return i + 1
		
	#create input for database
	@classmethod
	def create_EQ(cls, idx, atoms, X, E = None):
		eq = EQ(idx, atoms, X, E)
		res = [idx, Tool.ptoz(atoms), Tool.ptoz(X), E, 0, Tool.ptoz(eq.graph), eq.smiles]
		res = res
		return res

	#create task database
	@classmethod
	def create_tasks(cls, jobname, idx, atoms):
		conn = sqlite3.connect(jobname + ".db")
		cur = conn.cursor()
	
		cur.execute("select graph from EQs where idx == ?", [idx])
		G = Tool.ztop(cur.fetchall()[0][0])

		cur.execute('CREATE TABLE EQ' + str(idx) + '(idx INTEGER, info BLOB, result BLOB, state INTEGER, Thash TEXT, time REAL)')
		for i in range(len(atoms)):
			for j in range(i+1, len(atoms)):
				th1 = Tool.Thash(idx, i, j, "+")
				th2 = Tool.Thash(idx, i, j, "-")

				cur.execute('INSERT INTO EQ' + str(idx) + ' values(?, ?, ?, ?, ?, ?)', [-1, Tool.ptoz([i, j , "+"]), Tool.ptoz(None), 0, th1, 0])
				cur.execute('INSERT INTO EQ' + str(idx) + ' values(?, ?, ?, ?, ?, ?)', [-1, Tool.ptoz([i, j , "-"]), Tool.ptoz(None), 0, th2, 0])
		conn.commit()
		conn.close()

	#match task information
	@classmethod
	def match_job_info(cls, i1, i2):
		return i1[0] == i2[0] and i1[1] == i2[1] and i1[2] == i2[2]

def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None
