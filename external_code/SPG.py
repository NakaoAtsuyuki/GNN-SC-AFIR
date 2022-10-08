#!/home/common_data/buchi/anaconda3/bin/python
##!/home/pyenv/shims/python
import os
import shutil
import glob
import sys
import sqlite3
import traceback

import numpy as np
import Tool

ror = {"plus":"+", "minus":"-", "global":"+"}

def check(a, b):
	return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

#get jobname
comfile = glob.glob("*.com")
comfile = [comfile[i] for i in np.argsort([len(s) for s in comfile])]
jobname = comfile[0].split(".")[0]

num = int(sys.argv[1])

while 1:
	try:
		conn = sqlite3.connect(jobname + ".db")
		cur = conn.cursor()

		cur.execute("select * from EQ" + str(num))
		tasks = cur.fetchall()
		thashs = [t[4] for t in tasks]

		f = open(jobname + "_EQ" + str(num) + "_iPATH.rrm")
		ro = {"plus":"+", "minus":"-", "global":"+"}
		cands = []
		for line in f:
			tmp = line[:-1].split(" ")
			cands.append([int(tmp[0]), int(tmp[1]), ro[tmp[2]]])
		f.close()

		shutil.copy(jobname + "_EQ" + str(num) + "_iPATH.rrm", jobname + "_EQ" + str(num) + "_icPATH.rrm")
		
		fr = open(jobname + "_EQ" + str(num) + "_icPATH.rrm")
		fw = open(jobname + "_EQ" + str(num) + "_hPATH.rrm", "w")
		for line in fr:
			tmp = line[:-1].split(" ")
			tmp = [int(tmp[0]), int(tmp[1]), ror[tmp[2]]]
			h = Tool.Thash(num, tmp[0], tmp[1], tmp[2])
			fw.write(h + "\n")
		fr.close()
		fw.close()

		f = open(jobname + "_EQ" + str(num) + "_pPATH.rrm")
		infos = []
		vals = []
		for line in f:
			tmp = line[:-1].split(":")
			infos.append(eval(tmp[0]))
			vals.append(float(tmp[1]))

		f = open(jobname + "_EQ" + str(num) + "_hPATH.rrm")
		vals = Tool.read_hPATH(vals, f, jobname, num)
		target = int(np.argmax(vals))
		target_task = infos[target]

		ro = {"+" : "plus", "-":"minus"}
		max_val = np.max(vals)

		vals[target] = 0
		next_target = np.argmax(vals)
		next_task = infos[next_target]

		f = open(jobname + "_EQ" + str(num) + "_tPATH.rrm", "w")
		f.write(str(target_task[0]) + " " +  str(target_task[1]) + " " + ro[target_task[2]] + "\n")
		f.close()

		shutil.copy(jobname + "_EQ" + str(num) + "_tPATH.rrm", jobname + "_EQ" + str(num) + "_iPATH.rrm")
		os.remove(jobname + "_EQ" + str(num) + "_iPATH.rrm")
		f = open(jobname + "_EQ" + str(num) + "_iPATH.rrm", "a")
		f.write(str(next_task[0]) + " " +  str(next_task[1]) + " " + ro[next_task[2]] + "\n")
		f.close()

		f = open(jobname + "_order.log", "a")
		f.write(str(num) + ":" + str(target) + ":"  + str(target_task) + ":" + str(max_val) + "\n")
		f.close()

		f = open(jobname + "_thash.log", "a")
		f.write(thashs[target] + "\n")
		f.close()

		f = open(str(num) + "_" + str(target), "w")
		f.close()
		break

	except:
		traceback.print_exc()
		with open(jobname +"_Ext_ERROR.log", "a") as f:
			traceback.print_exc(file = f)
			f.write("-SPG------------------------------\n")	

		continue
	finally:
		conn.close()
