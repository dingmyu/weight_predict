#coding:utf-8
import pandas as pd
import numpy as np
import os

data_path = '/root/Desktop/medicalData/'

feature_path = '/root/Desktop/feature/'
if not os.path.isdir(feature_path):
	os.mkdir(feature_path)
file_list = os.listdir(data_path)
file_list.sort() 

def ge_feature(mother_age,mount,preg_week,HC,AC,FL,AMN,SD,PI,BPD,heart_frequency,FL_BPD,HC_AC,FL_AC,idx_lat, birth_week):
	lat_feature = np.array([mount[idx_lat],preg_week[idx_lat],HC[idx_lat],AC[idx_lat],FL[idx_lat],AMN[idx_lat],SD[idx_lat],PI[idx_lat],BPD[idx_lat],heart_frequency[idx_lat],FL_BPD[idx_lat],HC_AC[idx_lat],FL_AC[idx_lat],birth_week - preg_week[idx_lat]], dtype = float)
	return np.concatenate([[mother_age[idx_lat]], lat_feature]).tolist()
	
'''def ge_feature(mother_age,mount,preg_week,HC,AC,FL,AMN,SD,PI,heart_frequency,idx_lat, idx_pre ,birth_week):
	
	lat_feature = np.array([mount[idx_lat],preg_week[idx_lat],HC[idx_lat],AC[idx_lat],FL[idx_lat],AMN[idx_lat],SD[idx_lat],PI[idx_lat]], dtype = float)
	pre_feature = np.array([mount[idx_pre],preg_week[idx_pre],HC[idx_pre],AC[idx_pre],FL[idx_pre],AMN[idx_pre],SD[idx_pre],PI[idx_pre]], dtype = float)
	
	new_feature = (lat_feature - pre_feature) / (preg_week[idx_lat] - preg_week[idx_pre]) * (birth_week - preg_week[idx_lat])
	return np.concatenate([[mother_age[idx_lat]], lat_feature, new_feature]).tolist()'''


if __name__ == '__main__':

	Y = []
	X = []
	birth_data = []
	count_num_2 = 0
	
	for every_file in file_list:
		
		if os.path.splitext(every_file)[1] == '.txt' and (not os.path.splitext(every_file)[0] == 'readme'):
			myfile = open(data_path + every_file)
			text = myfile.readlines()
			lines = len(text)
			for line in text:
				words = line.split('\t')
				birth_week = int(words[3])
				birth_data.append([[words[2],words[3],words[5]]])
				birth_weight = float(words[5])
				break
			if int(birth_weight) > 0:
				if lines > 1:
					
					df = pd.read_csv(data_path + every_file, header = None, names = ['mother_name', 'B_idx', 'mother_age', 'mount', 'BPD', 'preg_week', 'HC', 'AC', 'FL', 'AMN', 'SD', 'PI', 'heart_frequency', 'weights'], sep = '\t', skiprows = 1) 
					mother_age = df['mother_age'].get_values()
					mount = df['mount'].get_values()
					preg_week = df['preg_week'].get_values()
					HC = df['HC'].get_values()
					AC = df['AC'].get_values()
					FL = df['FL'].get_values()
					AMN = df['AMN'].get_values()
					SD = df['SD'].get_values()
					PI = df['PI'].get_values()
					BPD = df['BPD'].get_values()
					print every_file
					print BPD
					print FL
					FL_BPD = FL/BPD
					HC_AC = HC/AC
					FL_AC = FL/AC
					heart_frequency = df['heart_frequency'].get_values()
					idx_nonempty = np.where(preg_week > 0)
					num_nonempty = len(idx_nonempty[0])
					if num_nonempty >= 1:#有一行B超值即可
						sort_idx = np.argsort(preg_week[idx_nonempty])
						count_num_2 += 1
						X.append(ge_feature(mother_age,mount,preg_week,HC,AC,FL,AMN,SD,PI,BPD,heart_frequency,FL_BPD,HC_AC,FL_AC,idx_nonempty[0][sort_idx[-1]],birth_week))
						
						Y.append(birth_weight)
	X = np.array(X)
	Y = np.array(Y)
	for i in range(X.shape[1]):
		X[:,i] = X[:,i]/max(abs(X[:,i]))
		
	print X
	print count_num_2
	print np.where(np.isnan(X))
	np.savez(feature_path + 'data_1.npz',X = X, Y = Y)
