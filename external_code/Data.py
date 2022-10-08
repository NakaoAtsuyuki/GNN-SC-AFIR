import re
import glob
import sqlite3
import numpy as np

import EQ
import Tool

#The class for reading GRRM outputs
class Data:

	def __init__(self, jobname):
		self.jobname = jobname

		fn = jobname + ".db"
		conn = sqlite3.connect(fn)

		#Database for summary of search state
		cur = conn.cursor()
		cur.execute('CREATE TABLE Main(jobname STRING, graph_group_member BLOB, numb BLOB)')
		cur.execute('INSERT INTO Main values(?, ?, ?)', [jobname, Tool.ptoz([]), Tool.ptoz({})])

		#Database for information on each EQ
		cur.execute('CREATE TABLE EQs(idx INTEGER PRIMARY KEY, atoms BLOB, X BLOB, E REAL, reload_time REAL, graph BLOB, smiles STRING)')	
		conn.commit()
		conn.close()

	#Reading GRRM outputs
	def reload(self):
		#Reading jobname_EQ_list.log
		self.read_EQ_list()
		#Reading jobname_P*_EQ_numb.rrm
		self.read_numb()
		
		#Reading files for each EQ
		conn = sqlite3.connect(self.jobname + ".db")
		cur = conn.cursor()
		cur.execute("select numb from Main")
		numb = Tool.ztop(cur.fetchall()[0][0])
		conn.close()
		for eq in range(self.EQnum()):
			EQ.EQ.reload(self.jobname, eq, numb)
			
	#Reading jobname_EQ_list.log
	def read_EQ_list(self):
		file_name = r"./" + self.jobname + "_EQ_list.log"
		f = open(file_name, "r")
		
		count = 0
		mode = 0
		regex_float = re.compile('-*\d+.\d+')
		
		alist = []
		poslist = []

		for line in f:
			if len(line) == 0:
				continue
				
			if mode == 1 and "Energy" in line:
				if count >= self.EQnum(): 
					self.addEQ(alist, np.asarray(poslist), float(regex_float.findall(line)[1]))
					
				mode = 0
				count += 1
				
			if mode == 1:
				elements = line.split('\t')
				alist.append(elements[0].replace(' ', ''))
				poslist.append([float(e.replace(' ', '')) for e in elements[1:]])
				
			if mode == 0 and line[0] == "#":
				mode = 1
				alist = []
				poslist = []
				
		f.close()
	
	#Reading jobname_P*_EQ_numb.rrm
	def read_numb(self):
		filelist = glob.glob(self.jobname + "_P*_EQ_numb.rrm")
		res = {}
		regex_int = re.compile('\d+')
		
		for file in filelist:
			i = int(regex_int.findall(file)[-1])
			tmp = {}
			
			f = open(file, "r")
			for line in f:
				info = re.findall(r"[-+]?\d*\.\d+|\d+", line)
				tmp[int(info[0])] = int(info[1])
			f.close()
			
			res[i] = tmp
		
		conn = sqlite3.connect(self.jobname + ".db")
		cur = conn.cursor()
		cur.execute("update Main set numb = ?", (Tool.ptoz(res),))
		conn.commit()
		conn.close()
	
	#Getting the number of EQs
	def EQnum(self):
		conn = sqlite3.connect(self.jobname + ".db")
		cur = conn.cursor()
		cur.execute("select count(*) from EQs")
		result = cur.fetchall()[0][0]
		conn.close()
		return result
	
	#Add EQ information to database
	def addEQ(self, atoms, X, E = None):
		conn = sqlite3.connect(self.jobname + ".db")
		cur = conn.cursor()
	
		idx = self.EQnum()
		sql = EQ.EQ.create_EQ(idx, atoms, X, E)
		cur.execute('INSERT INTO EQs values(?, ?, ?, ?, ?, ?, ?)', sql)
		conn.commit()
		conn.close()
		
		sql = EQ.EQ.create_tasks(self.jobname, idx, atoms)
		self.make_group(EQ.EQ(idx, atoms, X, E))

	#Classifying an EQ by 2D structure
	def make_group(self, eq):
		num = eq.idx
		
		conn = sqlite3.connect(self.jobname + ".db")
		cur = conn.cursor()
		cur.execute("select graph_group_member from Main")
		graph_group_member = Tool.ztop(cur.fetchall()[0][0])
		
		new = True
		for g in graph_group_member:
			if eq.smiles == g["smiles"]:
				g["member"].append(num)
				new = False
				break
				
		if new:
			graph_group_member.append({"graph":eq.graph, "member":[num], "smiles":eq.smiles})
		cur.execute("update Main set graph_group_member = ?", (Tool.ptoz(graph_group_member),))
		conn.commit()
		conn.close()
		
	#Getting index of 2D structure groups which an EQ belongs to
	@classmethod
	def get_group(cls, G, num):
		for i in range(len(G)):
			if num in G[i]["member"]:
				return i
			
	#Getting AFIR result
	def get_result(self, job):
		return self.EQs[job[0]].get_result(job[1])
