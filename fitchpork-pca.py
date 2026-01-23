########################################################################
# Imports
####################################

# ML imports
import tensorflow as tf
import keras
from keras import layers

from scripts import WMSE, InversePCA

# other imports
import pandas as pd
import numpy as np
import os

########################################################################
# Load Data
####################################

# get slurm job ID:
job_ID = os.getenv("SLURM_JOB_ID")

# def data dir
repo_dir = '/rds/homes/o/oxs235/daviesgr-pcann/repos_data/ojscutt/fitchpork/'

data_dir = repo_dir + 'data/'

# load in train and validation
df_train = pd.read_hdf(data_dir+'bob_train_slim.h5', key='df')
df_val = pd.read_hdf(data_dir+'bob_val_slim.h5', key='df')

# load in PCA variables
pca_mean = np.loadtxt(data_dir+'pca_mean_fp.csv')
pca_components = np.loadtxt(data_dir+'pca_components_fp.csv')

# load in WMSE weights
WMSE_weights = np.loadtxt(data_dir+'WMSE_weights_fp.csv').tolist()

########################################################################
# Define Inputs and Outputs
####################################

# define inputs
inputs = [
    'log_initial_mass_std', 
    'log_initial_Zinit_std', 
    'log_initial_Yinit_std', 
    'log_initial_MLT_std', 
    'log_star_age_std'
]

# define outputs
classical_outputs = [
    'log_radius_std', 
    'log_luminosity_std', 
    'star_feh_std'
]

astero_outputs = [f'log_nu_0_{i+1}_std' for i in range(5,40)]

outputs = classical_outputs+astero_outputs

########################################################################
# Train
####################################

## model_name
model_name = 'fitchpork-pca' + job_ID

tb_dir = repo_dir + 'logs/fit/'
model_dir = repo_dir + 'models/'

## architecture variables
stem_d_layers = 2
stem_d_units = 128

ctine_d_layers = 2
ctine_d_units = 64

atine_d_layers = 6
atine_d_units = 128

initial_lr = 0.001


tf.keras.backend.clear_session()
######## stem
#### input
stem_input = keras.Input(shape=(len(inputs),))

#### dense layers
for stem_d_layer in range(stem_d_layers):
    if stem_d_layer == 0:
        stem = layers.Dense(stem_d_units, activation='elu')(stem_input)
    else:
        stem = layers.Dense(stem_d_units, activation='elu')(stem)

######## classical tine
#### dense layers
for ctine_d_layer in range(ctine_d_layers):
    if ctine_d_layer == 0:
        ctine = layers.Dense(ctine_d_units, activation='elu')(stem)
    else:
        ctine = layers.Dense(ctine_d_units, activation='elu')(ctine)

#### output
ctine_out = layers.Dense(
    len(classical_outputs), 
    name='classical_outs'
)(ctine)


######## astero tine
#### dense layers
for atine_d_layer in range(atine_d_layers):
    if atine_d_layer == 0:
        atine = layers.Dense(atine_d_units, activation='elu')(stem)
    else:
        atine = layers.Dense(atine_d_units, activation='elu')(atine)

#### output
atine = layers.Dense(int(len(pca_components)))(atine)
atine_out = InversePCA(
    pca_comps = pca_components, 
    pca_mean = pca_mean, 
    name='asteroseismic_outs'
)(atine)

######## construct and fit
model = keras.Model(
    inputs=stem_input, 
    outputs=[ctine_out, atine_out], 
    name=model_name
)

#### compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
  
model.compile(
    loss=[WMSE(WMSE_weights[:3]), 
          WMSE(WMSE_weights[3:])], 
    optimizer=optimizer
)

#### fit model
def scheduler(epoch, lr):
    if lr < 1e-5:
        return lr
    else:
        return lr * float(tf.math.exp(-0.00006))

lr_callback = tf.keras.callbacks.LearningRateScheduler(
    scheduler,
    verbose=0)
                                                   
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_dir + model_name + ".h5",
    monitor= 'val_loss',
    save_best_only= True,
    save_freq='epoch'
)    

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir+model_name) 

history = model.fit(df_train[inputs],
                    [
                        df_train[classical_outputs],
                        df_train[astero_outputs]
                    ],
                    validation_data=(
                        df_val[inputs],
                        [df_val[classical_outputs], 
                         df_val[astero_outputs]]
                    ),
                    batch_size=32768,
                    verbose=0,
                    epochs=100000,
                    callbacks=[lr_callback, cp_callback, tb_callback],
                    shuffle=True
                   )

########################################################################