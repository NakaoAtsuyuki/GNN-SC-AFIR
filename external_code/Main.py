import glob
import time
import subprocess
import os
import numpy as np
import Data

#Getting job name
comfile = glob.glob("*.com")
comfile = [comfile[i] for i in np.argsort([len(s) for s in comfile])]
jobname = comfile[0].split(".")[0]

#Initializing database
d = Data.Data(jobname)

starttime = time.time()
while 1:
	time.sleep(1)
	
	try:
		#Reading GRRM outputs
		d.reload()

		#Stopping criterion by calculation time
		stoptime = 60*60*24*365
		if time.time() - os.path.getmtime(jobname + ".csh") > stoptime:
			f = open(jobname + "_message_STOP.rrm", "w")
			f.close()

		#Stopping criterion by the number of AFIR-path calculations
		limitnum = 6000
		linenum = np.sum([1 for _ in open(jobname + "_order.log")])
		if linenum > limitnum:
			f = open(jobname + "_message_STOP.rrm", "w")
			f.close()
	except:
		pass
