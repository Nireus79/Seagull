from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from sklearn.preprocessing import MinMaxScaler
from toolbox import rescaler
from data_forming import events_data

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from keras import Input, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import GRU, Dense
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error
from data_forming import events_data

# https://github.com/ydataai/ydata-synthetic/
# https://github.com/ydataai/ydata-synthetic/blob/dev/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb

'''
Define Model hyperparameters
Networks:
Generator
Discriminator
Embedder
Recovery Network
TimeGAN is a Generative model based on RNN networks. In this package the implemented version follows a very simple
architecture that is shared by the four elements of the GAN.
Similarly to other parameters, the architectures of each element should be optimized and tailored to the data.
'''

# Specific to TimeGANs
seq_len = 24
n_seq = 88  # number of columns
hidden_dim = 24
gamma = 1

noise_dim = 32
dim = 128
batch_size = 128

log_step = 100
learning_rate = 5e-4
# For quick prototyping epochs = 10
epochs = 50

synth_data_len = 2000

gan_args = ModelParameters(
    batch_size=batch_size, lr=learning_rate, noise_dim=noise_dim, layers_dim=dim
)

train_args = TrainParameters(
    epochs=epochs, sequence_length=seq_len, number_sequences=n_seq
)


# data_path = 'csv/tb/ETHEUR_30m.csv'
stock_data = events_data  # pd.read_csv(data_path)
# stock_data = stock_data.drop(columns=['time'])
# stock_data['time'] = 1
# stock_data['time'] = stock_data.apply(lambda x: 1800000 * x.name if x.name != 0 else 0, axis=1)
# stock_data.time = pd.to_datetime(stock_data.time, unit='ms')
# stock_data.set_index('time', inplace=True)

cols = list(stock_data.columns)
col_len = len(stock_data.columns)
print('stock_data.shape')
print(stock_data.shape)
print('stock_data')
print(stock_data)

# Training the TimeGAN synthesizer

if path.exists("synthesizer_stock.pkl"):
    synth = TimeSeriesSynthesizer.load("synthesizer_stock.pkl")
else:
    synth = TimeSeriesSynthesizer(modelname="timegan", model_parameters=gan_args)
    synth.fit(stock_data, train_args, num_cols=cols)
    synth.save("synthesizer_stock.pkl")

# stock_data_blocks = processed_stock(path=data_path, seq_len=seq_len)
# synth_data = np.asarray(synth.sample(len(stock_data_blocks)))
synth_data = np.asarray(synth.sample(synth_data_len))
# (synth_data_len, 24, col_len) to (synth_data_len, col_len) shape

synth_data = synth_data.reshape(synth_data.shape[0], -1)[:, :col_len]

print('synth_data.shape')
print(synth_data.shape)
print('synth_data')
print(synth_data)

synthesized_df = pd.DataFrame(synth_data, columns=stock_data.columns)
print(synthesized_df)

synthesized_df.to_csv('synthesized_events')

# stock_data.plot()
# df.plot()
# plt.show()

# Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
# axes = axes.flatten()
#
# time = list(range(1, 25))
# obs = np.random.randint(len(stock_data_blocks))
#
# for j, col in enumerate(cols):
#     df = pd.DataFrame({'Real': stock_data_blocks[obs][:, j],
#                        'Synthetic': synth_data[obs][:, j]})
#     df.plot(ax=axes[j],
#             title=col,
#             secondary_y='Synthetic data', style=['-', '--'])
# fig.tight_layout()
#
# # Evaluation of the generated synthetic data (PCA and TSNE)
#
# sample_size = 250
# idx = np.random.permutation(len(stock_data_blocks))[:sample_size]
#
# real_sample = np.asarray(stock_data_blocks)[idx]
# synthetic_sample = np.asarray(synth_data)[idx]
#
# # For the purpose of comparison we need the data to be 2-Dimensional.
# # For that reason we are going to use only two componentes for both the PCA and TSNE.
# synth_data_reduced = real_sample.reshape(-1, seq_len)
# stock_data_reduced = np.asarray(synthetic_sample).reshape(-1, seq_len)
#
# n_components = 2
# pca = PCA(n_components=n_components)
# tsne = TSNE(n_components=n_components, n_iter=300)
#
# # The fit of the methods must be done only using the real sequential data
# pca.fit(stock_data_reduced)
#
# pca_real = pd.DataFrame(pca.transform(stock_data_reduced))
# pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))
#
# data_reduced = np.concatenate((stock_data_reduced, synth_data_reduced), axis=0)
# tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))
# # The scatter plots for PCA and TSNE methods
#
#
# fig = plt.figure(constrained_layout=True, figsize=(20, 10))
# spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
#
# # TSNE scatter plot
# ax = fig.add_subplot(spec[0, 0])
# ax.set_title('PCA results',
#              fontsize=20,
#              color='red',
#              pad=10)
#
# # PCA scatter plot
# plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:, 1].values,
#             c='black', alpha=0.2, label='Original')
# plt.scatter(pca_synth.iloc[:, 0], pca_synth.iloc[:, 1],
#             c='red', alpha=0.2, label='Synthetic')
# ax.legend()
#
# ax2 = fig.add_subplot(spec[0, 1])
# ax2.set_title('TSNE results',
#               fontsize=20,
#               color='red',
#               pad=10)
#
# plt.scatter(tsne_results.iloc[:sample_size, 0].values, tsne_results.iloc[:sample_size, 1].values,
#             c='black', alpha=0.2, label='Original')
# plt.scatter(tsne_results.iloc[sample_size:, 0], tsne_results.iloc[sample_size:, 1],
#             c='red', alpha=0.2, label='Synthetic')
#
# ax2.legend()
#
# fig.suptitle('Validating synthetic vs real data diversity and distributions',
#              fontsize=16,
#              color='grey')
#
#
# # Train synthetic test real (TSTR)
# # First implement a simple RNN model for prediction
# def RNN_regression(units):
#     opt = Adam(name='AdamOpt')
#     loss = MeanAbsoluteError(name='MAE')
#     model = Sequential()
#     model.add(GRU(units=units,
#                   name=f'RNN_1'))
#     model.add(Dense(units=6,
#                     activation='sigmoid',
#                     name='OUT'))
#     model.compile(optimizer=opt, loss=loss)
#     return model
#
#
# # Prepare the dataset for the regression model
# stock_data = np.asarray(stock_data_blocks)
# synth_data = synth_data[:len(stock_data)]
# n_events = len(stock_data)
#
# # Split data on train and test
# idx = np.arange(n_events)
# n_train = int(.75 * n_events)
# train_idx = idx[:n_train]
# test_idx = idx[n_train:]
#
# # Define the X for synthetic and real data
# X_stock_train = stock_data[train_idx, :seq_len - 1, :]
# X_synth_train = synth_data[train_idx, :seq_len - 1, :]
#
# X_stock_test = stock_data[test_idx, :seq_len - 1, :]
# y_stock_test = stock_data[test_idx, -1, :]
#
# # Define the y for synthetic and real datasets
# y_stock_train = stock_data[train_idx, -1, :]
# y_synth_train = synth_data[train_idx, -1, :]
#
# print('Synthetic X train: {}'.format(X_synth_train.shape))
# print('Real X train: {}'.format(X_stock_train.shape))
#
# print('Synthetic y train: {}'.format(y_synth_train.shape))
# print('Real y train: {}'.format(y_stock_train.shape))
#
# print('Real X test: {}'.format(X_stock_test.shape))
# print('Real y test: {}'.format(y_stock_test.shape))
#
# # Training the model with the real train data
# ts_real = RNN_regression(12)
# early_stopping = EarlyStopping(monitor='val_loss')
#
# real_train = ts_real.fit(x=X_stock_train,
#                          y=y_stock_train,
#                          validation_data=(X_stock_test, y_stock_test),
#                          epochs=200,
#                          batch_size=128,
#                          callbacks=[early_stopping])
#
# # Training the model with the synthetic data
# ts_synth = RNN_regression(12)
# synth_train = ts_synth.fit(x=X_synth_train,
#                            y=y_synth_train,
#                            validation_data=(X_stock_test, y_stock_test),
#                            epochs=200,
#                            batch_size=128,
#                            callbacks=[early_stopping])
#
# real_predictions = ts_real.predict(X_stock_test)
# synth_predictions = ts_synth.predict(X_stock_test)
#
# metrics_dict = {'r2': [r2_score(y_stock_test, real_predictions),
#                        r2_score(y_stock_test, synth_predictions)],
#                 'MAE': [mean_absolute_error(y_stock_test, real_predictions),
#                         mean_absolute_error(y_stock_test, synth_predictions)],
#                 'MRLE': [mean_squared_log_error(y_stock_test, real_predictions),
#                          mean_squared_log_error(y_stock_test, synth_predictions)]}
#
# results = pd.DataFrame(metrics_dict, index=['Real', 'Synthetic'])
#
# print('results')
# print(results)
# plt.show()
