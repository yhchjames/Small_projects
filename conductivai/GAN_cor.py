#%%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

#%%
# Clean data

data = pd.read_csv("test_assignment_sim.csv")
site_co = pd.read_csv("site_coordinates.csv")

sites_datas = []
for i in range(49):
    slice = data.iloc[:,[0,1,2,3]]
    slice['thickness'] = data.iloc[:,i+4]
    slice['SITE'] = i # in lgb, catagory should be a int
    slice['S_X'] = site_co['SITE_X'][i]
    slice['S_Y'] = site_co['SITE_Y'][i]
    sites_datas.append(slice)

site_thickness = sites_datas[0]
for j in range(1,49):
    site_thickness = pd.concat([site_thickness,sites_datas[j]])

site_thickness.reset_index(drop=True)
site_thickness['TOOL'] = site_thickness['TOOL'].astype('category')
site_thickness.describe()
thickness_dmy = pd.get_dummies(site_thickness)

clist = list(thickness_dmy.columns)
clist.pop(4)
clist.append(clist.pop(3))

thickness_dmy = thickness_dmy[clist]
thickness_dmy.info()

dftrain,dftest = train_test_split(thickness_dmy,test_size=0.1)

#normallize data
mean = dftrain.mean()
std = dftrain.std()
train_data = (dftrain - mean) / std
train_data=train_data.astype('float32')
train_data.info()


BATCH_SIZE = 30
noise_dim = 3

ts_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(len(dftrain)).batch(BATCH_SIZE)


#%%
def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(18, use_bias=False, input_shape=(12,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(36))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(36))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(18))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))
    return model

def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Dense(20, use_bias=False, input_shape=(10,)))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(40))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(20))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))
    return model


generator = make_generator_model()
discriminator = make_discriminator_model()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(real_data):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    features = tf.slice(real_data,[0,0],[BATCH_SIZE,9])
    thickness = tf.slice(real_data,[0,9],[BATCH_SIZE,1])
    noise_and_features = tf.concat([noise,features],1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_thickness = generator(noise_and_features, training=True)
        gethickness_and_features = tf.concat([features,generated_thickness],1)

        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(gethickness_and_features, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    
    if(os.path.isfile("GAN-1.h5")):
        os.remove('GAN-1.h5')
    

    for epoch in range(epochs):
        start = time.time()

        for data_batch in dataset:
            train_step(data_batch)
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    generator.save('GAN-1.h5')

# %%
train(ts_data,1000)
# %%

ganmodel = keras.models.load_model('GAN-1.h5',compile=False)

y_test = np.array(dftest['thickness'])

test_data = (dftest - mean) / std
testnoise = pd.DataFrame(np.random.randn(len(y_test), 3), columns=['n_1','n_2','n_3'])
noise_testdata = pd.concat([testnoise,test_data.reset_index(drop=True)],axis=1)

# testing data prediction
y_pred = ganmodel.predict(np.array(noise_testdata.drop('thickness', axis='columns')))
# reversing-normalization of y_pred
y_pred = np.reshape(y_pred * std['thickness'] + mean['thickness'], y_test.shape)

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)