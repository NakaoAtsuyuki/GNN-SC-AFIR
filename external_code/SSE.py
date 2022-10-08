#!/home/common_data/buchi/anaconda3/bin/python
##!/home/pyenv/shims/python
import time
import sys
import os
import Tool
sys.path.remove(os.getcwd())

import glob
import subprocess
import traceback
import shutil
import numpy as np
import sqlite3

ro3 = {"+":"plus", "-":"minus"}

#write jobname_EQPREF.rrm
def makeEQPREF(jobname, next_eq):
	fname_eqpref = jobname + "_EQPREF.rrm"
	fdat_eqpref_old = open(fname_eqpref, "r")
	fdat_eqpref_new = open(fname_eqpref+".rw", "w")

	## write first line
	line = fdat_eqpref_old.readline()
	neq = int(line.split()[3])

	## write after second line
	fdat_eqpref_new.write("# of EQs: %d\n" % neq)
	for i, line in enumerate(fdat_eqpref_old):
		tmp_data=line.split()
		tag_judge_tobeapplied=1#int(tmp_da
		if i == next_eq:
			priority_value=1
		else:
			priority_value=0
			tag_judge_tobeapplied=-1
		fdat_eqpref_new.write("%3d %18.12f\n" % (tag_judge_tobeapplied, priority_value))
	
	fdat_eqpref_old.close()
	fdat_eqpref_new.close()

	cmd_cp = "cp %s %s" % (fname_eqpref+".rw", fname_eqpref)
	subprocess.call(cmd_cp, shell = True)

#job名の取得
comfile = glob.glob("*.com")
comfile = [comfile[i] for i in np.argsort([len(s) for s in comfile])]
jobname = comfile[0].split(".")[0]

if not os.path.exists(jobname + ".db"):
	subprocess.Popen(["nohup", "python", "Main.py", "&"])
	subprocess.Popen(["nohup", "python", "Training.py", "&"])
	subprocess.Popen(["nohup", "python", "Pred.py", "&"])

cmd_cp = "cp %s %s" % (jobname + "_EQPREF.rrm", jobname + "_EQPREF.rrm.cp")
subprocess.call(cmd_cp, shell = True)

#EQ selection
while 1:
	try:
		fname_eqpref = jobname + "_EQPREF.rrm"
		fdat_eqpref = open(fname_eqpref, "r")
		fdat_eqpref.readline()

		rprior = []
		for i, line in enumerate(fdat_eqpref):
			tmp= [t for t in  line[:-1].split(" ") if len(t) > 0]
			rprior.append(float(tmp[1]))
		fdat_eqpref.close()

		finished = [1 for i in range(len(rprior))]
		if os.path.exists(jobname + "_finishEQ.rrm"):
			fname_eqpref = jobname + "_finishEQ.rrm"
			fdat_eqpref = open(fname_eqpref, "r")

			for i, line in enumerate(fdat_eqpref):
				finished[i] = float(line[:-1])

			fdat_eqpref.close()

		prior = [p*f+1e-300 for p, f in zip(rprior, finished)]
		break

	except:
		continue

while 1:
	try:
		#prediction and task selection
		target_EQ = int(np.argmax(prior))
		try:
			conn = sqlite3.connect(jobname + ".db")
			cur = conn.cursor()
			cur.execute("select * from EQ" + str(target_EQ))
			tasks = cur.fetchall()
			a = tasks[1]
		except:
			prior[target_EQ] = 1e-300*np.random.rand()
			continue

		f = open(jobname + "_SSE.rrm", "w")
		f.write(str(target_EQ) + "\n")
		f.close()
		while not os.path.exists(jobname + "_prediction.rrm"):
			time.sleep(0.01)

		while 1:
			f = open(jobname + "_prediction.rrm")
			infos = []
			vals = []
			ghashs = []
			thashs = []
			hvals = []
			comp = False
			for line in f:
				if "END" in line:
					comp = True
					break	
				tmp = line[:-1].split(":")
				infos.append(eval(tmp[0]))
				vals.append(float(tmp[1]))	
				thashs.append(tmp[2])
				hvals.append(int(tmp[3]))

			if comp:
				break

		os.remove(jobname + "_prediction.rrm")

		if len(vals) == 0:
			prior[target_EQ] = 1e-300*np.random.rand()
			continue

		makeEQPREF(jobname, target_EQ)

		#write files
		if os.path.exists(jobname + "_EQ" + str(target_EQ) + "_tPATH.rrm"):
			f = open(jobname + "_EQ" + str(target_EQ) + "_hPATH.rrm")
			vals = Tool.read_hPATH(vals, f, jobname, target_EQ)

			f = open(jobname + "_EQ" + str(target_EQ) + "_tPATH.rrm", "a")
			target = int(np.argmax(vals))
			f.write(str(infos[target][0]) + " " +  str(infos[target][1]) + " " + ro3[infos[target][2]] + "\n")
			f.close()

			shutil.copy(jobname + "_EQ" + str(target_EQ) + "_tPATH.rrm", jobname + "_EQ" + str(target_EQ) + "_iPATH.rrm")
			f = open(jobname + "_EQ" + str(target_EQ) + "_iPATH.rrm", "a")
			f.write(str(infos[target][0]) + " " +  str(infos[target][1]) + " " + ro3[infos[target][2]] + "\n")
			f.close()

			f = open(jobname + "_EQ" + str(target_EQ) + "_cPATH.rrm", "a")
			num = np.sum([1 for _ in open(jobname + "_EQ" + str(target_EQ) + "_cPATH.rrm")])
			f.write("PATH-" + str(num) + " start: Third order coef. =     " + str(num*1.0) + "\n")
			f.close()

			linenum = np.sum([1 for _ in open(jobname + "_order.log")])
			f = open(jobname + "_order.log", "a")
			f.write(str(linenum) + "/" + str(target_EQ) + ":" + str(target) + ":" + str(infos[target]) + ":" + str(vals[target])+ "/" + str(rprior[target_EQ]) + ":" + str(finished[target_EQ]) + ":" + str(prior[target_EQ]) + "\n")
			f.close()

			f = open(jobname + "_thash.log", "a")
			f.write(thashs[target] + "\n")
			f.close()
			
			finished[target_EQ] = 1 if np.min(hvals) < 1000 else 0

			f = open(str(target_EQ) + "_" + str(target), "w")
			f.close()
		else:
			f = open(jobname + "_EQ" + str(target_EQ) + "_pPATH.rrm", "w")
			for v, info in zip(vals, infos):
				f.write(str(info) + ":" + str(v) + "\n")
			f.close()
		
		f = open(jobname + "_finishEQ.rrm", "w")
		for v in finished:
			f.write(str(v) + "\n")
		f.close()

		break
	
	except:
		prior[target_EQ] = 1e-300*np.random.rand()
		traceback.print_exc()
		with open(jobname +"_Ext_ERROR.log", "a") as f:
			traceback.print_exc(file = f)
			f.write("-SSE-----------------------------\n")

		conn.close()
		continue
