#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import os

data_path = '/root/Desktop/medicalData/'

feature_path = '/root/Desktop/feature/'
if not os.path.isdir(feature_path):
	os.mkdir(feature_path)
file_list = os.listdir(data_path)
file_list.sort() 

def ge_feature(mother_age,mount,preg_week,HC,AC,FL,AMN,SD,PI,BPD,heart_frequency,FL_BPD,HC_AC,FL_AC,idx_lat,birth_week):
	lat_feature = np.array([mount[idx_lat],preg_week[idx_lat],HC[idx_lat],AC[idx_lat],FL[idx_lat],AMN[idx_lat],SD[idx_lat],PI[idx_lat],BPD[idx_lat],heart_frequency[idx_lat],FL_BPD[idx_lat],HC_AC[idx_lat],FL_AC[idx_lat]], dtype = float)
	#pre_feature = np.array([mount[idx_pre],preg_week[idx_pre],HC[idx_pre],AC[idx_pre],FL[idx_pre],AMN[idx_pre],SD[idx_pre],PI[idx_pre],BPD[idx_pre],heart_frequency[idx_pre],FL_BPD[idx_pre],HC_AC[idx_pre],FL_AC[idx_pre]], dtype = float)
	detat = (birth_week - preg_week[idx_lat])/preg_week[idx_lat]
	#new_feature = (lat_feature - pre_feature) / (preg_week[idx_lat] - preg_week[idx_pre]) * (birth_week - preg_week[idx_lat])
	return np.concatenate([[detat],lat_feature]).tolist()
	
'''def ge_feature(mother_age,mount,preg_week,HC,AC,FL,AMN,SD,PI,heart_frequency,idx_lat, idx_pre ,birth_week):
	
	lat_feature = np.array([mount[idx_lat],preg_week[idx_lat],HC[idx_lat],AC[idx_lat],FL[idx_lat],AMN[idx_lat],SD[idx_lat],PI[idx_lat]], dtype = float)
	pre_feature = np.array([mount[idx_pre],preg_week[idx_pre],HC[idx_pre],AC[idx_pre],FL[idx_pre],AMN[idx_pre],SD[idx_pre],PI[idx_pre]], dtype = float)
	
	new_feature = (lat_feature - pre_feature) / (preg_week[idx_lat] - preg_week[idx_pre]) * (birth_week - preg_week[idx_lat])
	return np.concatenate([[mother_age[idx_lat]], lat_feature, new_feature]).tolist()'''


if __name__ == '__main__':#代码重用

	Y = []
	X = []
	birth_data = []
	count_num_2 = 0
	
	for every_file in file_list:
		
		if os.path.splitext(every_file)[1] == '.txt' and (not os.path.splitext(every_file)[0] == 'readme'):#os.path.splitext分离文件名和扩展名，保留txt的而且名字不是readme的
			myfile = open(data_path + every_file)
			text = myfile.readlines()#每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型
			lines = len(text)
			for line in text:#每一行处理
				words = line.split('\t')#\t  以制表符分隔
				birth_week = int(words[3])#妊娠周数
				birth_data.append([[words[2],words[3],words[5]]])#母亲年龄，妊娠周数，婴儿体重，貌似没卵用
				birth_weight = float(words[5])#婴儿体重
				break
			if int(birth_weight) > 0:
				if lines > 1:
					
					df = pd.read_csv(data_path + every_file, header = None, names = ['mother_name', 'B_idx', 'mother_age', 'mount', 'BPD', 'preg_week', 'HC', 'AC', 'FL', 'AMN', 'SD', 'PI', 'heart_frequency', 'weights'], sep = '\t', skiprows = 1) #母亲姓名 B超号 母亲年龄 胎盘厚度  BPD 临床孕周 HC AC FL AMN 脐动脉S/D PI 胎心率 预测体重，跳过一行
					mother_age = df['mother_age'].get_values()
					mount = df['mount'].get_values()
					preg_week = df['preg_week'].get_values()
					HC = df['HC'].get_values()
					AC = df['AC'].get_values()
					FL = df['FL'].get_values()
					AMN = df['AMN'].get_values()
					SD = df['SD'].get_values()
					PI = df['PI'].get_values()#把除了第一行的都转换成矩阵
					BPD = df['BPD'].get_values()#读出来的都是数字
					print every_file
					print BPD
					print FL
					FL_BPD = FL/BPD
					HC_AC = HC/AC
					FL_AC = FL/AC
					heart_frequency = df['heart_frequency'].get_values()
					idx_nonempty = np.where(preg_week > 0)#找到临床孕周大于0的
					num_nonempty = len(idx_nonempty[0])
					if num_nonempty >= 1:
						sort_idx = np.argsort(preg_week[idx_nonempty])#按照临床孕周索引排序
						count_num_2 += 1
					#	if count_num_2 == 251 or count_num_2 == 433:
							#print preg_week[idx_nonempty[0][sort_idx[-1]]] - preg_week[idx_nonempty[0][sort_idx[-2]]]#只利用最后两次数据
							#print ge_feature(mother_age,mount,preg_week,HC,AC,FL,AMN,SD,PI,BPD,heart_frequency,FL_BPD,HC_AC,FL_AC,idx_nonempty[0][sort_idx[-1]],idx_nonempty[0][sort_idx[-2]],birth_week)
							#print every_file
						X.append(ge_feature(mother_age,mount,preg_week,HC,AC,FL,AMN,SD,PI,BPD,heart_frequency,FL_BPD,HC_AC,FL_AC,idx_nonempty[0][sort_idx[-1]],birth_week))
						
						Y.append(birth_weight)
	X = np.array(X)#函数返回值，母亲年龄，和后两次数据生成的两列数据
	Y = np.array(Y)#体重
	for i in range(X.shape[1]):#矩阵的列数
		X[:,i] = X[:,i]/max(abs(X[:,i]))#除以最大的绝对值   归一化
		
	print X
	print count_num_2#输出大于两行的条目数
	print np.where(np.isnan(X))#nan为缺失值
	np.savez(feature_path + 'data2.npz',X = X, Y = Y)
