#encoding:utf-8
import numpy as np
import scipy as sp
import pandas as pd

from sklearn import metrics


#生成提交格式的文件
def create_submission(ids, predictions, filename='submission.csv'):
    submissions = np.concatenate((ids.reshape(len(ids), 1), predictions.reshape(len(predictions), 1)), axis=1)
    df = DataFrame(submissions)
    df.to_csv(filename, header=['id', 'click'], index=False)

#metrics原生log loss计算方法
def log_loss(y_true,y_pred):
	result=metrics.log_loss(y_true,y_pred)
	return result

#腾讯提供log loss计算方法
def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll


#合并其他特征到train文件
def merge_feature_to_train(train_data):
	"""
	加载train数据文件，并且合并除了user_app_actions和user_installedapps特征的其他特征，生成all_data.csv文件

	Args:
		train_data: DataFrame格式的特征-标签数据
	Returns:None

	"""
	folder_path="../../dataset/pre/"
	files=['user.csv','position.csv','ad.csv','app_categories.csv']
	keys=['userID','positionID','creativeID','appID']
	for index,file in enumerate(files):
		right_data=pd.read_csv(folder_path+file)
		train_data=train_data.merge(right_data, left_on=keys[index], right_on=keys[index], how='left')
	train_data.to_csv('all_data.csv',header=0)

#统计特征与label的相关性
def feature_label_correlation(data):
	"""
	统计特征与label的相关性，生成feature_label_correlation.txt文件

	Args:
		data: DataFrame格式的特征-标签数据
	Returns:None

	"""
	features=['creativeID', 'userID','positionID', 'connectionType', 'telecomsOperator', 'age', 'gender','education','marriageStatus', 'haveBaby', 'hometown', 'residence','sitesetID', 'positionType', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory']
	for feature in features:
		r=data.groupby(feature)['label'].mean()
		print(feature)
		print("\n")
		print(r)
		with open('feature_label_correlation.txt','a') as f:
			f.write(str(r))
			f.write("\n")
			f.write("==========s========================================")
			f.write("\n")

#分段clickTime并且统计每个段的概率，然后填充到新列clickTimePro中
def add_feature_clickTimePro(train_data):
	train_data['clickHour']=pd.Series([str(x)[2:4]for x in data.clickTime])
	print(train_data.tail(1))
	hourDict=train_data.groupby(['clickHour'])['label'].mean()
	print(hourDict)
	train_data['clickTimePro']=0.0
	for i in hourDict.index:
		train_data.loc[train_data.clickHour==i,'clickTimePro']=hourDict[i]
	print(train_data.tail(3))
	return train_data
