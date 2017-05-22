# encoding:utf-8
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics


import data_processing as dp


# 生成提交格式的文件
def create_submission(ids, predictions, filename='submission.csv'):
    # submissions = np.concatenate((ids.reshape(len(ids), 1), predictions.reshape(len(predictions), 1)), axis=1)
    df = pd.DataFrame(ids)
    df['pro'] = pd.DataFrame(predictions)
    df.to_csv(filename, header=['id', 'pro'], index=False)


# metrics原生log loss计算方法
def log_loss(y_true, y_pred):
    result = metrics.log_loss(y_true, y_pred)
    return result


# 腾讯提供log loss计算方法
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


# 合并其他特征到train文件
def merge_feature_to_train(train_data):
    """
    加载train数据文件，并且合并除了user_app_actions和user_installedapps特征的其他特征，生成all_data.csv文件

    Args:
        train_data: DataFrame格式的特征-标签数据
    Returns:None

    """
    folder_path = "../../dataset/pre/"
    files = ['user.csv', 'position.csv', 'ad.csv', 'app_categories.csv']
    keys = ['userID', 'positionID', 'creativeID', 'appID']
    for index, file in enumerate(files):
        right_data = pd.read_csv(folder_path + file)
        train_data = train_data.merge(right_data, left_on=keys[index], right_on=keys[index], how='left')
    train_data.to_csv('all_train_data.csv')


# 统计特征与label的相关性
def feature_label_correlation(data):
    """
    统计特征与label的相关性，生成feature_label_correlation.txt文件

    Args:
        data: DataFrame格式的特征-标签数据
    Returns:None

    """
    features = ['creativeID', 'userID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
                'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType', 'adID',
                'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory']
    for feature in features:
        r = data.groupby(feature)['label'].mean()
        print(feature)
        print("\n")
        print(r)
        with open('feature_label_correlation.txt', 'a') as f:
            f.write(str(r))
            f.write("\n")
            f.write("==========s========================================")
            f.write("\n")


# 分段clickTime并且统计每个段的概率，然后填充到新列clickTimePro中
def add_pro_feature(data, flag='train'):
    train_data = None
    if flag == 'train':
        data = data.drop(['conversionTime', 'userID'], axis=1)

        data['clickHour'] = pd.Series([str(x)[2:4] for x in data.clickTime])
        hourDict = data.groupby(['clickHour'])['label'].mean()
        data['clickTimePro'] = 0.0
        for i in hourDict.index:
            data.loc[data.clickHour == i, 'clickTimePro'] = hourDict[i]
        data = data.drop(['clickHour', 'clickTime'], axis=1)
        print('clickTime to clickTimePro finished!')
        # data['conversionHour']=pd.Series([str(x)[2:4] for x in data.conversionTime if pd.isnull(x)==True])
        # hourDict=data.groupby(['conversionHour'])['label'].mean()
        # data['conversionTimePro']=0.0
        # for i in hourDict.index:
        # 	data.loc[data.conversionHour==i,'conversionTimePro']=hourDict[i]
        # data=data.drop(['conversionHour','conversionTime'],axis=1)



        positionIDDict = data.groupby(['positionID'])['label'].mean()
        data['positionIDPro'] = 0.0
        for i in positionIDDict.index:
            data.loc[data.positionID == i, 'positionIDPro'] = positionIDDict[i]
        data = data = data.drop(['positionID'], axis=1)
        print('positionID to positionIDPro finished!')

        connectionTypeDict = data.groupby(['connectionType'])['label'].mean()
        data['connectionTypePro'] = 0.0
        for i in connectionTypeDict.index:
            data.loc[data.connectionType == i, 'connectionTypePro'] = connectionTypeDict[i]
        data = data.drop(['connectionType'], axis=1)
        print('connectionType to connectionTypePro finished!')

        telecomsOperatorDict = data.groupby(['telecomsOperator'])['label'].mean()
        data['telecomsOperatorPro'] = 0.0
        for i in telecomsOperatorDict.index:
            data.loc[data.telecomsOperator == i, 'telecomsOperatorPro'] = telecomsOperatorDict[i]
        data = data.drop(['telecomsOperator'], axis=1)
        print('telecomsOperator to telecomsOperatorPro finished!')

        creativeIDDict = data.groupby(['creativeID'])['label'].mean()
        data['creativeIDPro'] = 0.0
        for i in creativeIDDict.index:
            data.loc[data.creativeID == i, 'creativeIDPro'] = creativeIDDict[i]
        data = data.drop(['creativeID'], axis=1)
        print('creativeID to creativeIDPro finished!')

    # userIDDict=data.groupby(['userID'])['label'].mean()
    # data['userIDPro']=0.0
    # for i in userIDDict.index:
    # 	data.loc[data.userID==i,'userIDPro']=userIDDict[i]
    # data=data.drop(['userID'],axis=1)
    else:
        train_data = dp.read_data('train.csv')
        data = data.drop(['userID'], axis=1)

        train_data['clickHour'] = pd.Series([str(x)[2:4] for x in train_data.clickTime])
        data['clickHour'] = pd.Series([str(x)[2:4] for x in data.clickTime])
        hourDict = train_data.groupby(['clickHour'])['label'].mean()
        data['clickTimePro'] = 0.0
        for i in hourDict.index:
            data.loc[data.clickHour == i, 'clickTimePro'] = hourDict[i]
        data = data.drop(['clickHour', 'clickTime'], axis=1)
        print('clickTime to clickTimePro finished!')
        # data['conversionHour']=pd.Series([str(x)[2:4] for x in data.conversionTime if pd.isnull(x)==True])
        # hourDict=data.groupby(['conversionHour'])['label'].mean()
        # data['conversionTimePro']=0.0
        # for i in hourDict.index:
        # 	data.loc[data.conversionHour==i,'conversionTimePro']=hourDict[i]
        # data=data.drop(['conversionHour','conversionTime'],axis=1)



        positionIDDict = train_data.groupby(['positionID'])['label'].mean()
        data['positionIDPro'] = 0.0
        for i in positionIDDict.index:
            data.loc[data.positionID == i, 'positionIDPro'] = positionIDDict[i]
        data = data.drop(['positionID'], axis=1)
        print('positionID to positionIDPro finished!')

        connectionTypeDict = train_data.groupby(['connectionType'])['label'].mean()
        data['connectionTypePro'] = 0.0
        for i in connectionTypeDict.index:
            data.loc[data.connectionType == i, 'connectionTypePro'] = connectionTypeDict[i]
        data = data.drop(['connectionType'], axis=1)
        print('connectionType to connectionTypePro finished!')

        telecomsOperatorDict = train_data.groupby(['telecomsOperator'])['label'].mean()
        data['telecomsOperatorPro'] = 0.0
        for i in telecomsOperatorDict.index:
            data.loc[data.telecomsOperator == i, 'telecomsOperatorPro'] = telecomsOperatorDict[i]
        data = data.drop(['telecomsOperator'], axis=1)
        print('telecomsOperator to telecomsOperatorPro finished!')

        creativeIDDict = train_data.groupby(['creativeID'])['label'].mean()
        data['creativeIDPro'] = 0.0
        for i in creativeIDDict.index:
            data.loc[data.creativeID == i, 'creativeIDPro'] = creativeIDDict[i]
        data = data.drop(['creativeID'], axis=1)
        print('creativeID to creativeIDPro finished!')

    # userIDDict=data.groupby(['userID'])['label'].mean()
    # data['userIDPro']=0.0
    # for i in userIDDict.index:
    # 	data.loc[data.userID==i,'userIDPro']=userIDDict[i]
    # data=data.drop(['userID'],axis=1)

    data.to_csv('new_' + flag + '.csv')  # train 13分钟 ；test 1分钟
    return data


# 统计colums_list里面对应信息的概率，并填充到相应的新列当中
def add_feature_colum(train_data, test_data, colunms_list):
    """
        从一个填充很多其他列信息的表中，挑选出我们需要
        进行训练的概率信息的feature

        Parameters
        --------------
        train_data: 填充了概率统计列后的DataFrame格式的train_merge
        test_data:  填充了概率统计列后的DataFrame格式的test_merge
        colunms_list:  选取作为特征提取的colunms的列表
        feature_list:  填充后的概率信息的后新的colunms name的列表

        Returns
        --------------
        train_feature: train_megre数据集中被筛选出来作为回归的X输入
        test_feature: test_megre数据中被筛选出来作为回归模型预测的对象
        
        Examples
        --------------
        >>>colunms_list = ['clickTime', 'creativeID', 'positionID', 'connectionType', 'telecomsOperator']
        >>>add_feature_colum(train_data, test_data, colunms_list)
        clickHour is a clickTime feature...
        clickHour feature finished!
        creativeID feature finished!
        positionID feature finished!
        connectionType feature finished!
        telecomsOperator feature finished!

    """
    for colunmName in colunms_list:

        newColunm = colunmName + 'Pro'
        if colunmName == 'clickTime':
            train_data['clickHour'] = pd.Series([str(x)[2:4] for x in train_data.clickTime])
            test_data['clickHour'] = pd.Series([str(x)[2:4] for x in test_data.clickTime])
            colunmName = 'clickHour'
            print(colunmName + ' is a clickTime feature...')

        feature_pro = train_data.groupby(colunmName, as_index=True)['label'].mean()
        feature_pro = feature_pro.rename(newColunm)
        feature_pro = pd.DataFrame(feature_pro)
        feature_pro.reset_index(level=0, inplace=True)
        train_data = train_data.merge(feature_pro, left_on=colunmName, right_on=colunmName, how='left')
        test_data = test_data.merge(feature_pro, left_on=colunmName, right_on=colunmName, how='left')

        print(colunmName + ' feature finished!')

    return train_data, test_data


def train_and_test(train_data, test_data):
    lr = LogisticRegression()

    train_data = train_data.drop(['index'], axis=1)
    y_train = train_data['label']
    x_train = train_data.drop(['label'], axis=1)

    x_test = test_data.drop(['index', 'instanceID', 'label'], axis=1)

    lr.fit(x_train, y_train)  # 训练模型耗时11s
    y_test_arr = lr.predict_proba(x_test)  # 进行预测
    y_test = y_test_arr[:, 1]

    ids = np.array(range(1, 338490))
    create_submission(ids, y_test)
    pass
