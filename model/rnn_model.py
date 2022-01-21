#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://blog.csdn.net/marsjhao/article/details/68490105 https://blog.csdn.net/u014281392/article/details/77103747
import numpy as np
import scipy.io as sio
np.random.seed(355)  # for reproducibility
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout, Bidirectional,Activation,Flatten,GRU,LSTM
from keras.layers import Dense, Input, TimeDistributed,RepeatVector
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import Model
from keras.optimizers import Adam
import math
import time
import config_APS_1 as cf
import six.moves.cPickle as pickle
from keras_self_attention import SeqSelfAttention
import keras


# In[2]:


# 全局变量
batch_size = 32
epochs = 20
CELL_SIZE=32
OUTPUT_SIZE=20
# input image dimensions
TIME_STEPS = cf.n_time_interval
datasetName =cf.datasetName
INPUT_SIZE = len(cf.degree_interval_list)+1
input_shape = (TIME_STEPS, INPUT_SIZE) 
print (input_shape)


# In[12]:


id_train, x_train, L, y_train, sz_train, time_train, vocabulary_size = pickle.load(open(cf.train_pkl, 'rb'))
id_test, x_test, L_test, y_test, sz_test, time_test, _ = pickle.load(open(cf.test_pkl, 'rb'))
id_val, x_val, L_val, y_val, sz_val, time_val, _ = pickle.load(open(cf.val_pkl, 'rb'))
for i in range(len(x_train)):
    x_train[i] = np.array(x_train[i])
for i in range(len(x_val)):
    x_val[i] = np.array(x_val[i])
for i in range(len(x_test)):
    x_test[i] = np.array(x_test[i])
x_val = np.array(x_val)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_val = np.array(y_val)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
print(y_train.shape)
# x_train = x_train.reshape(x_train.shape[0], TIME_STEPS, INPUT_SIZE, 1)
# x_test = x_test.reshape(x_test.shape[0], TIME_STEPS, INPUT_SIZE, 1)
# x_val = x_val.reshape(x_val.shape[0], TIME_STEPS, INPUT_SIZE, 1)


# In[13]:


train_ratio = 0.8
train_sz = x_train.shape[0]
train_sample = int (np.floor(train_sz*(1-train_ratio)))
print (train_sample)
x_train = x_train[train_sample:-1]
y_train = y_train[train_sample:-1]
print(x_train.shape)
print(y_train.shape)


# In[4]:


print(np.mean(y_train))
print(np.mean(y_test))
print(np.mean(y_test) - np.mean(y_train))
# x_train = x_train.resh


# In[14]:


## LSTM SelfAttention

model = Sequential()
model.add(keras.layers.Bidirectional(keras.layers.GRU(units=OUTPUT_SIZE, input_shape=input_shape,return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.GRU(units=OUTPUT_SIZE,return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
#model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Flatten()) #拉成一维数据
model.add(Dense(16, activation='relu')) 
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu')) #全连接层2
# model.add(Dense(1, activation='relu')) #全连接层2
 
#编译模型
model.compile(loss='mean_squared_error',optimizer='adadelta',metrics=['mse'])
#model.summary()  #打印模型

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,verbose = 1, factor=0.5, min_lr = 0.0000004)
#训练模型
f = open(str(train_ratio)+'result_epoch_rnn_SA.csv','w')
for epoch in range(epochs):
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_data=(x_val, y_val), 
              shuffle=True, verbose=1,callbacks=[learning_rate_reduction])
    test_score = model.evaluate(x_test, y_test, verbose=0)
    val_score  = model.evaluate(x_val, y_val, verbose=0)
    f.write(str(epoch)+',Test score,'+ str(test_score[0])+',Val_score,'+str(val_score[0])+'\n')
    print('Test score:', test_score[0])
f.close()


# In[5]:


## LSTM 
model = Sequential()
model.add(GRU(units = OUTPUT_SIZE, activation='tanh', input_shape=input_shape,return_sequences=True))
model.add(GRU(units = OUTPUT_SIZE, activation='tanh',return_sequences=True))
model.add(Flatten()) #拉成一维数据
model.add(Dense(16, activation='relu')) 
model.add(Dense(1, activation='relu')) #全连接层2
# model.add(Dense(1, activation='relu')) #全连接层2
 
#编译模型
model.compile(loss='mean_squared_error',optimizer='adadelta',metrics=['mse'])
model.summary()  #打印模型

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,verbose = 1, factor=0.5, min_lr = 0.000000001)
#训练模型
f = open(datasetName+'result_epoch_rnn.csv','w')
for epoch in range(epochs):
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_data=(x_val, y_val), 
              shuffle=True, verbose=1,callbacks=[learning_rate_reduction])
    test_score = model.evaluate(x_test, y_test, verbose=0)
    val_score  = model.evaluate(x_val, y_val, verbose=0)
    f.write(str(epoch)+',Test score,'+ str(test_score[0])+',Val_score,'+str(val_score[0])+'\n')
    print('Test score:', test_score[0])
f.close()


# In[7]:


score = model.evaluate(x_test, y_test, verbose=0)
y_test_pre = model.predict(x_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])
f = open(datasetName+'detail_result_rnn_.csv','w')
for i in range (len(y_test)):
    f.write(str(y_test[i])+','+str(y_test_pre[i][0])+'\n')
f.close()


# In[9]:


## LSTM autoencoder
retrunSeq=True
input_data = Input((None,INPUT_SIZE))
encoder=GRU(units=OUTPUT_SIZE*2,  activation='tanh',return_sequences=retrunSeq, name="gru1")(input_data)
encoder=GRU(units=OUTPUT_SIZE,  activation='tanh',return_sequences=retrunSeq, name="gru2")(encoder)
encoder_out=Dense(OUTPUT_SIZE,activation='tanh')(encoder)
encoder_model = Model(inputs=input_data, outputs=encoder_out)

decoder=GRU(units=OUTPUT_SIZE, activation='tanh',return_sequences=retrunSeq,name="de_gru1")(encoder_out)
decoder=GRU(units=OUTPUT_SIZE*2,  activation='tanh', return_sequences=retrunSeq, name="de_gru2")(decoder)
decoder_out=Dense(INPUT_SIZE,activation='relu')(decoder)
autoencoder=Model(input_data,decoder_out)
 
#编译模型
autoencoder.compile(loss='mean_squared_error',optimizer='adadelta',metrics=['mse'])
autoencoder.summary()  #打印模型

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,verbose = 1, factor=0.5, min_lr = 0.000001)
#训练模型
autoencoder.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, x_val), 
          shuffle=True, verbose=1,callbacks=[learning_rate_reduction])
#预测模型
encoded_latent_train = encoder_model.predict(x_train)
encoded_latent_test = encoder_model.predict(x_test)
encoded_latent_val = encoder_model.predict(x_val)
if retrunSeq:
    encoded_latent_train=encoded_latent_train.reshape(-1,OUTPUT_SIZE*TIME_STEPS)
    encoded_latent_test=encoded_latent_test.reshape(-1,OUTPUT_SIZE*TIME_STEPS)
    encoded_latent_val=encoded_latent_val.reshape(-1,OUTPUT_SIZE*TIME_STEPS)

input_representation = Input(shape=(OUTPUT_SIZE*TIME_STEPS,))
latent_vector = Dense(16, activation='relu')(input_representation)
preddiction = Dense(1, activation='relu')(latent_vector)
model2 = Model(input= input_representation,output=preddiction)
#编译
model2.compile(loss='mean_squared_error',optimizer='adadelta',metrics=['mse'])
model2.summary()  #打印模型

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,verbose = 1, factor=0.5, min_lr = 0.00001)
#训练模型
f = open('result_epoch_rnn_ae.csv','w')
for epoch in range(epochs):
    model2.fit(encoded_latent_train, y_train, batch_size=batch_size, nb_epoch=1, validation_data=(encoded_latent_val, y_val), 
              shuffle=True, verbose=1,callbacks=[learning_rate_reduction])
    test_score = model2.evaluate(encoded_latent_test, y_test, verbose=0)
    val_score  = model2.evaluate(encoded_latent_val, y_val, verbose=0)
    f.write(str(epoch)+',Test score,'+ str(test_score[0])+',Val_score,'+str(val_score[0])+'\n')
    print('Test score:', test_score[0])
f.close()


# In[4]:





# In[ ]:


## LSTM Bi
model = Sequential()
#model.add(keras.layers.Embedding(input_dim=INPUT_SIZE, output_dim=16,mask_zero=True))
model.add(keras.layers.Bidirectional(keras.layers.GRU(units=OUTPUT_SIZE, input_shape=input_shape,return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.GRU(units=OUTPUT_SIZE,return_sequences=True)))

model.add(Flatten()) #拉成一维数据
model.add(Dense(16, activation='relu')) 
model.add(Dense(1, activation='relu')) #全连接层2
# model.add(Dense(1, activation='relu')) #全连接层2
 
#编译模型
model.compile(loss='mean_squared_error',optimizer='adadelta',metrics=['mse'])
#model.summary()  #打印模型

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,verbose = 1, factor=0.5, min_lr = 0.00000001)
#训练模型
f = open(datasetName+'result_epoch_rnn_bi.csv','w')
for epoch in range(epochs):
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_data=(x_val, y_val), 
              shuffle=True, verbose=1,callbacks=[learning_rate_reduction])
    test_score = model.evaluate(x_test, y_test, verbose=0)
    val_score  = model.evaluate(x_val, y_val, verbose=0)
    f.write(str(epoch)+',Test score,'+ str(test_score[0])+',Val_score,'+str(val_score[0])+'\n')
    print('Test score:', test_score[0])
f.close()


# In[6]:


## mlp
model = Sequential()
model.add(Flatten()) #拉成一维数据
model.add(Dense(32, activation='relu')) 
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu')) #全连接层2
# model.add(Dense(1, activation='relu')) #全连接层2
 
#编译模型
model.compile(loss='mean_squared_error',optimizer='adadelta',metrics=['mse'])
#model.summary()  #打印模型

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,verbose = 1, factor=0.5, min_lr = 0.000000005)
#训练模型
f = open(datasetName+'result_epoch_mlp.csv','w')
for epoch in range(epochs):
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_data=(x_val, y_val), 
              shuffle=True, verbose=1,callbacks=[learning_rate_reduction])
    test_score = model.evaluate(x_test, y_test, verbose=0)
    val_score  = model.evaluate(x_val, y_val, verbose=0)
    f.write(str(epoch)+',Test score,'+ str(test_score[0])+',Val_score,'+str(val_score[0])+'\n')
    print('Test score:', test_score[0])
f.close()
model.summary() 


# In[2]:


import config_hep as cf


# In[7]:


y_train[1]


# In[ ]:




