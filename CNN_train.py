# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:29:21 2017

@author: N.Chlis
"""

#from load_data import load_patients
#import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.layers import Dense, Activation
from keras.layers import BatchNormalization, Flatten
#from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Input
import h5py

def residual_block2D(input_tensor, block, filters = [128,128], kernel_size=3, dropout_rate_conv=0.0):
    """The identity block is the block that has no conv layer at shortcut.
    """
    conv_name = 'res' + str(block) + '_branch'
    bn_name = 'bn' + str(block) + '_branch'
    drop_name = 'drop' + str(block) + '_branch'
    bn_axis=-1
    
    x = Conv2D(filters=filters[0], kernel_size=kernel_size, padding='same', name=conv_name + 'a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name + 'a')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate_conv,name=drop_name + 'a')(x)
    
    x = Conv2D(filters=filters[1], kernel_size=kernel_size, padding='same', name=conv_name + 'b')(x)
    x = layers.add([x, input_tensor])
    x = BatchNormalization(axis=bn_axis, name=bn_name + 'b')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate_conv,name=drop_name + 'b')(x)
    
    return x

def residual_block2D_conv1(input_tensor, block, filters = [128,128], kernel_size=3, dropout_rate_conv=0.0):
    """The identity block is the block that has no conv layer at shortcut.
    """
    conv_name = 'res' + str(block) + '_branch'
    bn_name = 'bn' + str(block) + '_branch'
    drop_name = 'drop' + str(block) + '_branch'
    bn_axis=-1
    
    y = Conv2D(filters=filters[0], kernel_size=1, padding='same', name=conv_name + 'c1d')(input_tensor)
    x = Conv2D(filters=filters[0], kernel_size=kernel_size, padding='same', name=conv_name + 'a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name + 'a')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate_conv,name=drop_name + 'a')(x)
    
    x = Conv2D(filters=filters[1], kernel_size=kernel_size, padding='same', name=conv_name + 'b')(x)
    x = layers.add([x, y])
    x = BatchNormalization(axis=bn_axis, name=bn_name + 'b')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate_conv,name=drop_name + 'b')(x)
    
    return x
#%%
h5f = h5py.File('X_fpaul_exp2.h5','r')
#h5f.visit(print)#print all filenames in the file
X = h5f['X'][()]
#filenames = h5f['fnames'][()]
cd34 = h5f['intensity_CD34'][()]
fcgr = h5f['intensity_FcgR'][()]
obj_id = h5f['obj_id'][()].astype('int')
h5f.close()

#split into training and validation
(ix_tr, ix_val) = train_test_split(np.arange(len(X)), test_size=0.1, random_state=0)

X_tr = X[ix_tr,:,:,:]
X_val = X[ix_val,:,:,:]
X_tr = X_tr[:,:,:,0]#only BF channel
X_val = X_val[:,:,:,0]#only BF channel
#only 1 channel, need to reshape
X_tr = X_tr.reshape(X_tr.shape+(1,))
X_val = X_val.reshape(X_val.shape+(1,))

obj_id_tr = obj_id[ix_tr]
obj_id_val = obj_id[ix_val]

cd34_tr = cd34[ix_tr]
cd34_val = cd34[ix_val]

fcgr_tr = fcgr[ix_tr]
fcgr_val = fcgr[ix_val]

#data augmentation on train set
X_tr = np.concatenate((X_tr,np.flip(X_tr,axis=1),np.flip(X_tr,axis=2),
                      np.flip(np.flip(X_tr,axis=1),axis=2)),axis=0)

cd34_tr = np.concatenate((cd34_tr,cd34_tr,cd34_tr,cd34_tr),axis=0)
fcgr_tr = np.concatenate((fcgr_tr,fcgr_tr,fcgr_tr,fcgr_tr),axis=0)

#%%
dropout_rate=0.0
inner_activ = 'relu'
bnorm_axis = -1
print('Build model...')
input_tensor = Input(shape=X_tr.shape[1:], name='input_tensor')

#encoder
x = Conv2D(filters=16, kernel_size=(3,3), padding='same')(input_tensor)
x = BatchNormalization(axis=bnorm_axis)(x)
x = Activation(inner_activ)(x)
x = Conv2D(filters=16, kernel_size=(3,3), padding='same')(x)
x = BatchNormalization(axis=bnorm_axis)(x)
x = Activation(inner_activ)(x)
x = MaxPooling2D((2, 2))(x)#1024x768x64

x = residual_block2D_conv1(x, 1, filters = [32,32], kernel_size=3, dropout_rate_conv=0.0)
x = residual_block2D(x, 2, filters = [32,32], kernel_size=3, dropout_rate_conv=0.0)
x = MaxPooling2D((2, 2))(x)#1024x768x64

x = residual_block2D_conv1(x, 3, filters = [64,64], kernel_size=3, dropout_rate_conv=0.0)
x = residual_block2D(x, 4, filters = [64,64], kernel_size=3, dropout_rate_conv=0.0)
x = MaxPooling2D((2, 2))(x)#1024x768x64

x = residual_block2D_conv1(x, 5, filters = [128,128], kernel_size=3, dropout_rate_conv=0.0)
x = residual_block2D(x, 6, filters = [128,128], kernel_size=3, dropout_rate_conv=0.0)
x = MaxPooling2D((2, 2))(x)#1024x768x64

#global average pooling
x = AveragePooling2D(pool_size=(int(x.shape[1]), int(x.shape[2])), strides=None, padding='valid', data_format=None)(x)
x = Flatten(name='encoder')(x)

out_cd34 = Dense(1)(x)
out_cd34 = Activation('linear',name='out_cd34')(out_cd34)
out_fcgr = Dense(1)(x)
out_fcgr = Activation('linear',name='out_fcgr')(out_fcgr)

model=Model(inputs=input_tensor,outputs=[out_cd34, out_fcgr])
model.compile(loss='mae', optimizer='adam')
model.summary()

#%% train the model

filepath="CNN" #to save the weights

checkpoint = ModelCheckpoint(filepath+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
csvlog = CSVLogger('./trained_models/'+filepath+'_train_log.csv',append=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

model.fit(x=X_tr, y=[cd34_tr, fcgr_tr], validation_data=(X_val,[cd34_val, fcgr_val]),
                 epochs=50, batch_size=64, verbose=2,
                 initial_epoch=0,callbacks=[checkpoint, csvlog, early_stopping])


