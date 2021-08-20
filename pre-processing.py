import pandas as pd
import math
import numpy as np
import warnings
import sklearn

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

import os
from time import time
from tensorflow import keras
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical

def removeDifficulty(df):
    df.drop(['difficulty'], axis=1, inplace=True)

def attackClassify(df):

    df.replace({
        'attack': {
        'back' : 'dos',
        'land' : 'dos',
        'neptune' : 'dos',
        'pod' : 'dos',
        'smurf' : 'dos',
        'teardrop' : 'dos',
        'udpstorm' : 'dos',
        'processtable' : 'dos',
        'mailbomb' : 'dos',
        'apache2': 'dos'}}, #apache2 was not in KDDTrain
        inplace=True)
    df.replace({
        'attack': {
        'ftp_write' : 'r2l', 
        'guess_passwd' : 'r2l',
        'imap' : 'r2l',
        'multihop' : 'r2l',
        'phf' : 'r2l',
        'spy' : 'r2l',
        'warezclient' : 'r2l',
        'warezmaster' : 'r2l',
        'snmpgetattack' : 'r2l',
        'named' : 'r2l',
        'xlock' : 'r2l',
        'xsnoop' : 'r2l',
        'sendmail' : 'r2l'}},
        inplace=True)
    df.replace({
        'attack': {
        'buffer_overflow' : 'u2r', 
        'loadmodule' : 'u2r',
        'perl' : 'u2r',
        'rootkit' : 'u2r',
        'xterm' : 'u2r',
        'ps' : 'u2r',
        'httptunnel' : 'u2r',
        'sqlattack' : 'u2r',
        'worm' : 'u2r',
        'snmpguess' : 'u2r'}},
        inplace=True)
    df.replace({
        'attack': {
        'ipsweep' : 'probe',
        'nmap' : 'probe',
        'portsweep' : 'probe',
        'satan' : 'probe',
        'saint' : 'probe',
        'mscan' : 'probe'}},
        inplace=True)
    return df
    
dict_data_types = {
        'duration': 'int64',#1
        'protocol_type': 'object',#2
        'service': 'object',#3
        'flag': 'object',#4
        # 'src_bytes': 'int32', #5
        # 'dst_bytes': 'int32', #6
        # 'land': 'int32', #7
        # 'wrong_fragment': 'int32', #8
        # 'urgent': 'int32', #9
        # 'hot': 'int32', #10
        # 'num_failed_logins': 'int32', #11
        # 'logged_in': 'int32', #12
        # 'num_compromised',#13
        # 'root_shell',#14
        # 'su_attempted',#15
        # 'num_root',#16
        # 'num_file_creations',#17
        # 'num_shells',#18
        # 'num_access_files',#19
        # 'num_outbound_cmds',#20
        # 'is_host_login',#21
        # 'is_guest_login',#22
        # 'count',#23
        # 'srv_count',#24
        # 'serror_rate', #25
        # 'srv_serror_rate',#26
        # 'rerror_rate',#27
        # 'srv_rerror_rate',#28
        # 'same_srv_rate',#29
        # 'diff_srv_rate',#30
        # 'srv_diff_host_rate',#31
        # 'dst_host_count',#32
        # 'dst_host_srv_count',#33
        # 'dst_host_same_srv_rate',#34
        # 'dst_host_diff_srv_rate',#35
        # 'dst_host_same_src_port_rate',#36
        # 'dst_host_srv_diff_host_rate',#37
        # 'dst_host_serror_rate',#38
        # 'dst_host_srv_serror_rate',#39
        # 'dst_host_rerror_rate',#40
        'attack': 'object'}#42
        # 'dst_host_srv_rerror_rate',#41
        

def oneHotEncode(train, test):
    print("One Hot Encoding")
    categorical_features = ['protocol_type', 'service', 'flag']

    #enc.fit(train)
    trainX = train.drop('attack',axis=1).copy()
    trainY = train[['attack']].copy()
    testX = test.drop('attack',axis=1).copy()
    testY = test[['attack']].copy() 

    print(trainY)

    trainX_object = trainX.select_dtypes('object')
    
    print(trainX_object)
    testX_object = testX.select_dtypes('object')
    x_ohe = OneHotEncoder(sparse=False)
    x_ohe.fit(trainX_object)
    
    trainX_codes = x_ohe.transform(trainX_object)
    
    x_feature_names = x_ohe.get_feature_names(categorical_features)
    
    print(testX_object.info())
    
    train_enc_X = pd.concat([trainX.select_dtypes(exclude='object'), 
               pd.DataFrame(trainX_codes,columns=x_feature_names)], axis=1)

    testX_codes = x_ohe.transform(testX_object)
    
    test_enc_X = pd.concat([testX.select_dtypes(exclude='object'), 
               pd.DataFrame(testX_codes,columns=x_feature_names)], axis=1)
        
    y_ohe = OneHotEncoder(sparse=False)
    y_ohe.fit(trainY)
    train_enc_y = y_ohe.transform(trainY)
    test_enc_y = y_ohe.transform(testY)
    
    print(x_ohe.categories_)
    
    print(y_ohe.categories_)

    print("Train")
    print(train_enc_X)
    print(test_enc_X)
    print("Test")
    print(train_enc_y)
    print(test_enc_y)
    return train_enc_X, train_enc_y, test_enc_X, test_enc_y

def logTransform(trainX, testX):

    for col in trainX:
        colmax = trainX[col].max()
        if colmax > 100:
            trainX[col] = trainX[col].apply(np.log1p)
            testX[col] = testX[col].apply(np.log1p)
    
    
    scaler = MinMaxScaler((0,1))
    scale_trainX = scaler.fit_transform(trainX)
    print(scaler.data_min_)
    print(scaler.data_max_)
    scale_testX = scaler.transform(testX)
    
    
    print(scale_trainX)
    print(scale_testX)
    
    x_feature_names = trainX.columns.values.tolist()
    
    norm_trainX = pd.DataFrame(scale_trainX,columns=x_feature_names)
    norm_testX = pd.DataFrame(scale_testX,columns=x_feature_names)
    
    print(norm_trainX)
    print(norm_testX)
    return norm_trainX, norm_testX

names = ['duration', #1
    'protocol_type', #2
    'service', #3
    'flag', #4
    'src_bytes', #5
    'dst_bytes', #6
    'land',#7
    'wrong_fragment',#8
    'urgent',#9
    'hot',#10
    'num_failed_logins',#11
    'logged_in',#12
    'num_compromised',#13
    'root_shell',#14
    'su_attempted',#15
    'num_root',#16
    'num_file_creations',#17
    'num_shells',#18
    'num_access_files',#19
    'num_outbound_cmds',#20
    'is_host_login',#21
    'is_guest_login',#22
    'count',#23
    'srv_count',#24
    'serror_rate',#25
    'srv_serror_rate',#26
    'rerror_rate',#27
    'srv_rerror_rate',#28
    'same_srv_rate',#29
    'diff_srv_rate',#30
    'srv_diff_host_rate',#31
    'dst_host_count',#32
    'dst_host_srv_count',#33
    'dst_host_same_srv_rate',#34
    'dst_host_diff_srv_rate',#35
    'dst_host_same_src_port_rate',#36
    'dst_host_srv_diff_host_rate',#37
    'dst_host_serror_rate',#38
    'dst_host_srv_serror_rate',#39
    'dst_host_rerror_rate',#40
    'dst_host_srv_rerror_rate',#41
    'attack',#42
    'difficulty']#43

train = pd.read_csv('KDDTrain+.txt', names=names, header=None, dtype=dict_data_types, index_col=False, low_memory = False)
test = pd.read_csv('KDDTest+.txt', names=names, header=None, dtype=dict_data_types, index_col=False, low_memory = False)

removeDifficulty(train)
removeDifficulty(test)

print(train.info())

trainReplaced = attackClassify(train)


testReplaced = attackClassify(test)

trainX1, trainY, testX1, testY = oneHotEncode(trainReplaced, testReplaced)

trainX2, testX2 = logTransform(trainX1, testX1)

train_X = np.asarray(trainX2.to_numpy())
train_y = np.asarray(trainY)
 
test_X = np.asarray(testX2.to_numpy())
test_y = np.asarray(testY)

print(test_X)
print(test_y)

# pd.DataFrame(test_X).iloc[:80].to_csv(path_or_buf='./k2c/simpleRNNIDS_train.csv', header=None, index=False)
# pd.DataFrame(test_y).iloc[:80].to_csv(path_or_buf='./k2c/simpleRNNIDS_test.csv', header=None, index=False)


pd.DataFrame(test_X).to_csv(path_or_buf='./k2c/simpleRNNIDS_train.csv', header=None, index=False)
pd.DataFrame(test_y).to_csv(path_or_buf='./k2c/simpleRNNIDS_test.csv', header=None, index=False)