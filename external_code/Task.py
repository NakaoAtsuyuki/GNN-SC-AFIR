import Tool
import time
import os

#The class for control result of task
class Task:
	
	def __init__(self, a1, a2, s, target):
		self.idx = -1
		self.info = [a1, a2, s]
		self.result = None
		self.state = 0
		
	#set a result of AFIR calculation
	@classmethod
	def set_result(cls, info, num, cur):
		if len(info) > 0:
			try:
				cur.execute("select rowid, state from EQ" + str(num) + " where idx = ?", (info["idx"], ))
				tmp = cur.fetchall()[0]
				rowid = tmp[0]-1
				state = tmp[1]
				if state == 2:
					return True

				res = [[r[0],r[1].E,r[2]] if r[0] == "EQ" else [r[0], r[1].E] for r in info["path"]]
				tmp = time.time()
				cur.execute("update EQ" + str(num) + " set result = ?, state = ?, time = ? where idx = ?", (Tool.ptoz(res), 2, tmp, info["idx"]))
				with open("result.log", "a") as f:
					f.write("num:" + str(num) + ", info:" + str(info["idx"]) + ", result:" + str(res) + ", time:" + str(tmp) + "\n")
				os.remove(str(num) + "_" + str(rowid))
				return True
			except:
				pass

		return False
