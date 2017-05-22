# encoding:utf-8
import numpy as np
import scipy as sp
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LogisticRegression


# 生成提交格式的文件
def create_submission(ids, predictions, filename='submission.csv'):
    df = pd.DataFrame(ids)
    df['pro'] = pd.DataFrame(predictions)
    df.to_csv(filename, header=['id', 'click'], index=False)


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
    # folder_path = "../../dataset/pre/"me
    files = ['user.csv', 'position.csv', 'ad.csv', 'app_categories.csv']
    keys = ['userID', 'positionID', 'creativeID', 'appID']
    for index, file in enumerate(files):
        right_data = pd.read_csv(file)
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


column_list = ['clickTime', 'creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'sitesetID',
               'positionType']
feature_list = ['clickTimePro', 'creativeIDPro', 'positionIDPro', 'connectionTypePro', 'telecomsOperatorPro',
                'sitesetIDPro', 'positionTypePro']


# 统计columns_list里面对应信息的概率，并填充到相应的新列当中
def filter_feature_column(train_data, test_data, column_list, feature_list):
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
        >>>columns_list = ['clickTime', 'creativeID', 'positionID', 'connectionType', 'telecomsOperator']
        >>>feature_list = ['clickTimePro', 'creativeIDPro', 'positionIDPro', 'connectionTypePro', 'telecomsOperatorPro', 'sitesetIDPro', 'positionTypePro']
        >>>filter_feature_column(train_data, test_data, colunms_list, feature_list)
            
            
    """
    for column_name in column_list:

        new_column = column_name + 'Pro'
        if column_name == 'clickTime':
            train_data['clickHour'] = pd.Series([str(x)[2:4] for x in train_data.clickTime])
            test_data['clickHour'] = pd.Series([str(x)[2:4] for x in test_data.clickTime])
            column_name = 'clickHour'
            print(column_name + ' is a clickTime feature...')

        feature_pro = train_data.groupby(column_name, as_index=True)['label'].mean()
        feature_pro = feature_pro.rename(new_column)
        feature_pro = pd.DataFrame(feature_pro)
        feature_pro.reset_index(level=0, inplace=True)
        train_data = train_data.merge(feature_pro, left_on=column_name, right_on=column_name, how='left')
        test_data = test_data.merge(feature_pro, left_on=column_name, right_on=column_name, how='left')

        print(column_name + ' feature finished!')

    return train_data[feature_list], test_data[feature_list]


def train_and_predict_test(train_feature, test_feature, y_label, write_sub=False):
    lr = LogisticRegression()

    lr.fit(train_feature, y_label)  # 训练模型耗时11s
    y_test_arr = lr.predict_proba(test_feature.fillna(0))  # 进行预测
    y_test = y_test_arr[:, 1]

    ids = np.array(range(1, 338490))
    if write_sub:
        create_submission(ids, y_test)
    pass


def construct_feature(train_data, test_data, columns_list, features_list):
    train_feature, test_feature = filter_feature_column(train_data, test_data, columns_list, features_list)
    y_label = train_data['label']
    train_and_predict_test(train_feature, test_feature, y_label)


def sampling_loss(train_data, columns_list, features_list, sampling_ratio=0.1):
    sampling_number = int(len(train_data) * sampling_ratio)
    rows = np.random.choice(train_data.index.values, sampling_number)
    sampling_data = train_data.ix[rows]
    train_feature, test_feature = filter_feature_column(train_data, sampling_data, columns_list, features_list)
    y_label = train_data['label']

    lr = LogisticRegression()

    lr.fit(train_feature, y_label)
    y_test_arr = lr.predict_proba(test_feature.fillna(0))
    y_pre = y_test_arr[:, 1]
    y_action = sampling_data['label']
    result = logloss(np.asarray(y_action), y_pre)
    return result
