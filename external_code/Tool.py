import numpy as np
import pickle
import bz2
import hashlib
import sqlite3
import os

PROTOCOL = pickle.HIGHEST_PROTOCOL
#translate object to binary
def ptoz(obj):
	return bz2.compress(pickle.dumps(obj, PROTOCOL), 3)

#translate binary to object
def ztop(b):
	return pickle.loads(bz2.decompress(b))

#traslate task to hash
def Thash(idx, i, j, ro):
	atom = sorted([i, j])
	return hashlib.md5(str(atom + [idx, ro]).encode()).hexdigest()

#make value of tasks excluded by SC algorithm 0
def read_hPATH(vals, f, jobname, num):
	conn = sqlite3.connect(jobname + ".db")
	cur = conn.cursor()

	cur.execute("select thash from EQ" + str(num))
	thash = [h[0] for h in cur.fetchall()]
	
	todo = []
	for line in f:
		todo += [line[:-1]]

	return [v if h in todo else 0 for v, h in zip(vals, thash)]

#create list for storing prediction result
def pred_base(jobname, num):
	conn = sqlite3.connect(jobname + ".db")
	cur = conn.cursor()
	cur.execute("select count(*) from EQ" + str(num))
	tnum = cur.fetchall()[0][0]

	cur.execute("select thash from EQ" + str(num))
	thash = [h[0] for h in cur.fetchall()]

	y = -np.ones([len(thash)])
	if not os.path.exists(jobname + "_EQ" + str(num) + "_hPATH.rrm"):
		return y*0

	f = open(jobname + "_EQ" + str(num) + "_hPATH.rrm")
	todo = []
	for line in f:
		todo += [line[:-1]]
	f.close()

	for i, t in enumerate(thash):
		if t in todo:
			y[i] = 0

	conn.close()
	return y
