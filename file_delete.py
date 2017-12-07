import os

def delete(dir):
	for k in os.listdir('./'):
		for i in os.listdir(k+'/'+i):
			print(i)
			if(i[0]=='d'):
				os.remove(i)