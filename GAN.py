import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as md

import torch

from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType

df = pd.read_csv('csv/tb/ETHEUR_5m.csv')
df.time = pd.to_datetime(df.time, unit='ms')
df.set_index('time', inplace=True)
df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
ohlc = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}
COLUMNS = ["Open", "High", "Low", "Close", 'Volume']

df = df.resample('30min').apply(ohlc)
print(df)
train_df = df[["datetime"] + COLUMNS]

for c in COLUMNS:
    plt.plot(train_df["datetime"], train_df[c], label=c)
plt.xticks(rotation=90)
plt.legend()
plt.ylabel("Temperature (Celsius)")
plt.xlabel("Date")
plt.show()

# DGAN needs many example time series to train. Split into 1-day slices to
# create multiple examples.
features = train_df.drop(columns="datetime").to_numpy()
# Obsevations every 10 minutes, so 144 * 10 minutes = 1 day
n = features.shape[0] // 144
features = features[:(n * 144), :].reshape(-1, 144, features.shape[1])
# Shape is now (# examples, # time points, # features)
print(features.shape)

# Show a few of the 1-day training samples
xaxis_1day = train_df["datetime"][0:144]


def plot_day(f):
    for i, c in enumerate(COLUMNS):
        plt.plot(xaxis_1day, f[:, i], label=c)
    ax = plt.gca()
    ax.xaxis.set_major_locator(md.HourLocator(byhour=range(2, 24, 3)))
    ax.xaxis.set_major_formatter(md.DateFormatter("%H:%M"))
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Temperature (Celsius)")
    plt.show()


plot_day(features[125, :, :])
plot_day(features[3, :, :])
plot_day(features[21, :, :])

# Recommended to train with a GPU
torch.cuda.is_available()

# Train DGAN model
model = DGAN(DGANConfig(
    max_sequence_len=features.shape[1],
    sample_len=12,
    batch_size=min(1000, features.shape[0]),
    apply_feature_scaling=True,
    apply_example_scaling=False,
    use_attribute_discriminator=False,
    generator_learning_rate=1e-4,
    discriminator_learning_rate=1e-4,
    epochs=10000,
))

model.train_numpy(
    features,
    feature_types=[OutputType.CONTINUOUS] * features.shape[2],
)

# Generate synthetic data
_, synthetic_features = model.generate_numpy(1000)
# Show some synthetic 1-day samples
plot_day(synthetic_features[825, :, :])
plot_day(synthetic_features[42, :, :])
plot_day(synthetic_features[496, :, :])

# Compare (non-temporal) correlations between the 4 temperatures
synthetic_df = pd.DataFrame(synthetic_features.reshape(-1, synthetic_features.shape[2]), columns=train_df.columns[1:])

print("Correlation in real data:")
print(train_df.corr())
print()
print("Correlation in synthetic data:")
print(synthetic_df.corr())

# Compare distribution of T_out values
plt.hist([features[:, :, 3].flatten(), synthetic_features[:, :, 3].flatten()],
         label=["real", "synthetic"],
         bins=25,
         density=True)
plt.legend()
plt.xlabel("Temperature (Celsius)")
plt.ylabel("Density")
plt.show()

# Compare distribution of 1-step (10 minute) diffs for T_out
real_diffs = np.diff(features, axis=1)
synthetic_diffs = np.diff(synthetic_features, axis=1)

plt.hist([real_diffs[:, :, 3].flatten(), synthetic_diffs[:, :, 3].flatten()],
         label=["real", "synthetic"],
         bins=50,
         density=True)
plt.legend()
plt.xlabel("10 minute temperature change")
plt.ylabel("Density")
plt.show()

# 1-step diffs for synthetic data have higher variance, this aligns with the
# increased noise visible in the plots

# Correlations between temperature variables are similar

# https://github.com/gretelai/gretel-synthetics
# https://github.com/gretelai/gretel-synthetics/blob/master/examples/timeseries_dgan.ipynb
# https://www.youtube.com/watch?v=Jpua18PKRGU


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from pathlib import Path
# from tqdm import tqdm
#
# from keras.models import Sequential, Model
# from keras.layers import GRU, Dense, RNN, GRUCell, Input
# from keras.losses import BinaryCrossentropy, MeanSquaredError
# from keras.optimizers import Adam
# from keras.callbacks import TensorBoard
# from keras.utils import plot_model
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
#
# # https://github.com/stefan-jansen/synthetic-data-for-finance/blob/main/01_TimeGAN_TF2.ipynb
# warnings.filterwarnings('ignore')
#
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# if gpu_devices:
#     print('Using GPU')
#     tf.config.experimental.set_memory_growth(gpu_devices[0], True)
# else:
#     print('Using CPU')
#
# sns.set_style('white')
# np.random.seed(42)
# tf.random.set_seed(1234)
#
# results_path = Path('time_gan')
# if not results_path.exists():
#     results_path.mkdir()
# experiment = 0
# log_dir = results_path / f'experiment_{experiment:02}'
# if not log_dir.exists():
#     log_dir.mkdir(parents=True)
# hdf_store = results_path / 'TimeSeriesGAN.h5'
#
# df = pd.read_csv('csv/tb/ETHEUR_5m.csv')
# # df = pd.read_csv('stocks.csv',
# #                  index_col='date',
# #                  parse_dates=['date'])
# print(df.info())
#
# seq_len = 24
# n_seq = 6
# batch_size = 128
#
# tickers = ['BA', 'CAT', 'DIS', 'GE', 'IBM', 'KO']
#
# axes = df.div(df.iloc[0]).plot()
# # subplots=True,
# #                                figsize=(14, 6),
# #                                layout=(3, 3),
# #                                title=tickers,
# #                                legend=False,
# #                                rot=0,
# #                                lw=1,
# #                                color='k'
# # for ax in axes.flatten():
# #     ax.set_xlabel('')
#
# plt.suptitle('Normalized Price Series')
# plt.gcf().tight_layout()
# sns.despine()
#
# sns.clustermap(df.corr(),
#                annot=True,
#                fmt='.2f',
#                cmap=sns.diverging_palette(h_neg=20,
#                                           h_pos=220), center=0)
#
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df).astype(np.float32)
#
# data = []
# for i in range(len(df) - seq_len):
#     data.append(scaled_data[i:i + seq_len])
#
# n_windows = len(data)
#
# real_series = (tf.data.Dataset
#                .from_tensor_slices(data)
#                .shuffle(buffer_size=n_windows)
#                .batch(batch_size))
# real_series_iter = iter(real_series.repeat())
#
#
# def make_random_data():
#     while True:
#         yield np.random.uniform(low=0, high=1, size=(seq_len, n_seq))
#
#
# random_series = iter(tf.data.Dataset
#                      .from_generator(make_random_data, output_types=tf.float32)
#                      .batch(batch_size)
#                      .repeat())
#
# hidden_dim = 24
# num_layers = 3
#
# writer = tf.summary.create_file_writer(log_dir.as_posix())
#
# X = input(shape=[seq_len, n_seq], name='RealData')
# Z = input(shape=[seq_len, n_seq], name='RandomData')
#
#
# def make_rnn(n_layers, hidden_units, output_units, name):
#     return Sequential([GRU(units=hidden_units,
#                            return_sequences=True,
#                            name=f'GRU_{i + 1}') for i in range(n_layers)] +
#                       [Dense(units=output_units,
#                              activation='sigmoid',
#                              name='OUT')], name=name)
#
#
# embedder = make_rnn(n_layers=3,
#                     hidden_units=hidden_dim,
#                     output_units=hidden_dim,
#                     name='Embedder')
# recovery = make_rnn(n_layers=3,
#                     hidden_units=hidden_dim,
#                     output_units=n_seq,
#                     name='Recovery')
#
# generator = make_rnn(n_layers=3,
#                      hidden_units=hidden_dim,
#                      output_units=hidden_dim,
#                      name='Generator')
# discriminator = make_rnn(n_layers=3,
#                          hidden_units=hidden_dim,
#                          output_units=1,
#                          name='Discriminator')
# supervisor = make_rnn(n_layers=2,
#                       hidden_units=hidden_dim,
#                       output_units=hidden_dim,
#                       name='Supervisor')
#
# train_steps = 10000
# gamma = 1
#
# mse = MeanSquaredError()
# bce = BinaryCrossentropy()
# H = embedder(X)
# X_tilde = recovery(H)
#
# autoencoder = Model(inputs=X,
#                     outputs=X_tilde,
#                     name='Autoencoder')
#
# print(autoencoder.summary())
#
# plot_model(autoencoder,
#            to_file=(results_path / 'autoencoder.png').as_posix(),
#            show_shapes=True)
#
# autoencoder_optimizer = Adam()
#
#
# @tf.function
# def train_autoencoder_init(x):
#     with tf.GradientTape() as tape:
#         x_tilde = autoencoder(x)
#         embedding_loss_t0 = mse(x, x_tilde)
#         e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)
#
#     var_list = embedder.trainable_variables + recovery.trainable_variables
#     gradients = tape.gradient(e_loss_0, var_list)
#     autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
#     return tf.sqrt(embedding_loss_t0)
#
#
# for step in tqdm(range(train_steps)):
#     X_ = next(real_series_iter)
#     step_e_loss_t0 = train_autoencoder_init(X_)
#     with writer.as_default():
#         tf.summary.scalar('Loss Autoencoder Init', step_e_loss_t0, step=step)
#
# supervisor_optimizer = Adam()
#
#
# @tf.function
# def train_supervisor(x):
#     with tf.GradientTape() as tape:
#         h = embedder(x)
#         h_hat_supervised = supervisor(h)
#         g_loss_s = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])
#
#     var_list = supervisor.trainable_variables
#     gradients = tape.gradient(g_loss_s, var_list)
#     supervisor_optimizer.apply_gradients(zip(gradients, var_list))
#     return g_loss_s
#
#
# for step in tqdm(range(train_steps)):
#     X_ = next(real_series_iter)
#     step_g_loss_s = train_supervisor(X_)
#     with writer.as_default():
#         tf.summary.scalar('Loss Generator Supervised Init', step_g_loss_s, step=step)
#
# E_hat = generator(Z)
# H_hat = supervisor(E_hat)
# Y_fake = discriminator(H_hat)
#
# adversarial_supervised = Model(inputs=Z,
#                                outputs=Y_fake,
#                                name='AdversarialNetSupervised')
#
# print(adversarial_supervised.summary())
#
# plot_model(adversarial_supervised, show_shapes=True)
#
# Y_fake_e = discriminator(E_hat)
#
# adversarial_emb = Model(inputs=Z,
#                         outputs=Y_fake_e,
#                         name='AdversarialNet')
#
# print(adversarial_emb.summary())
#
# plot_model(adversarial_emb, show_shapes=True)
#
# X_hat = recovery(H_hat)
# synthetic_data = Model(inputs=Z,
#                        outputs=X_hat,
#                        name='SyntheticData')
#
# print(synthetic_data.summary())
#
# print(
#     plot_model(synthetic_data, show_shapes=True))
#
#
# def get_generator_moment_loss(y_true, y_pred):
#     y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
#     y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
#     g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
#     g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
#     return g_loss_mean + g_loss_var
#
#
# Y_real = discriminator(H)
# discriminator_model = Model(inputs=X,
#                             outputs=Y_real,
#                             name='DiscriminatorReal')
#
# print(discriminator_model.summary())
#
# plot_model(discriminator_model, show_shapes=True)
# generator_optimizer = Adam()
# discriminator_optimizer = Adam()
# embedding_optimizer = Adam()
#
#
# @tf.function
# def train_generator(x, z):
#     with tf.GradientTape() as tape:
#         y_fake = adversarial_supervised(z)
#         generator_loss_unsupervised = bce(y_true=tf.ones_like(y_fake),
#                                           y_pred=y_fake)
#
#         y_fake_e = adversarial_emb(z)
#         generator_loss_unsupervised_e = bce(y_true=tf.ones_like(y_fake_e),
#                                             y_pred=y_fake_e)
#         h = embedder(x)
#         h_hat_supervised = supervisor(h)
#         generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])
#
#         x_hat = synthetic_data(z)
#         generator_moment_loss = get_generator_moment_loss(x, x_hat)
#
#         generator_loss = (generator_loss_unsupervised +
#                           generator_loss_unsupervised_e +
#                           100 * tf.sqrt(generator_loss_supervised) +
#                           100 * generator_moment_loss)
#
#     var_list = generator.trainable_variables + supervisor.trainable_variables
#     gradients = tape.gradient(generator_loss, var_list)
#     generator_optimizer.apply_gradients(zip(gradients, var_list))
#     return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss
#
#
# @tf.function
# def train_embedder(x):
#     with tf.GradientTape() as tape:
#         h = embedder(x)
#         h_hat_supervised = supervisor(h)
#         generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])
#
#         x_tilde = autoencoder(x)
#         embedding_loss_t0 = mse(x, x_tilde)
#         e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised
#
#     var_list = embedder.trainable_variables + recovery.trainable_variables
#     gradients = tape.gradient(e_loss, var_list)
#     embedding_optimizer.apply_gradients(zip(gradients, var_list))
#     return tf.sqrt(embedding_loss_t0)
#
#
# @tf.function
# def get_discriminator_loss(x, z):
#     y_real = discriminator_model(x)
#     discriminator_loss_real = bce(y_true=tf.ones_like(y_real),
#                                   y_pred=y_real)
#
#     y_fake = adversarial_supervised(z)
#     discriminator_loss_fake = bce(y_true=tf.zeros_like(y_fake),
#                                   y_pred=y_fake)
#
#     y_fake_e = adversarial_emb(z)
#     discriminator_loss_fake_e = bce(y_true=tf.zeros_like(y_fake_e),
#                                     y_pred=y_fake_e)
#     return (discriminator_loss_real +
#             discriminator_loss_fake +
#             gamma * discriminator_loss_fake_e)
#
#
# @tf.function
# def train_discriminator(x, z):
#     with tf.GradientTape() as tape:
#         discriminator_loss = get_discriminator_loss(x, z)
#
#     var_list = discriminator.trainable_variables
#     gradients = tape.gradient(discriminator_loss, var_list)
#     discriminator_optimizer.apply_gradients(zip(gradients, var_list))
#     return discriminator_loss
#
#
# step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
# for step in range(train_steps):
#     # Train generator (twice as often as discriminator)
#     for kk in range(2):
#         X_ = next(real_series_iter)
#         Z_ = next(random_series)
#
#         # Train generator
#         step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(X_, Z_)
#         # Train embedder
#         step_e_loss_t0 = train_embedder(X_)
#
#     X_ = next(real_series_iter)
#     Z_ = next(random_series)
#     step_d_loss = get_discriminator_loss(X_, Z_)
#     if step_d_loss > 0.15:
#         step_d_loss = train_discriminator(X_, Z_)
#
#     if step % 1000 == 0:
#         print(f'{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
#               f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')
#
#     with writer.as_default():
#         tf.summary.scalar('G Loss S', step_g_loss_s, step=step)
#         tf.summary.scalar('G Loss U', step_g_loss_u, step=step)
#         tf.summary.scalar('G Loss V', step_g_loss_v, step=step)
#         tf.summary.scalar('E Loss T0', step_e_loss_t0, step=step)
#         tf.summary.scalar('D Loss', step_d_loss, step=step)
#
# # synthetic_data.save(log_dir / 'synthetic_data')
#
# generated_data = []
# for i in range(int(n_windows / batch_size)):
#     Z_ = next(random_series)
#     d = synthetic_data(Z_)
#     generated_data.append(d)
#
# print(len(generated_data))
#
# generated_data = np.array(np.vstack(generated_data))
# print(generated_data.shape)
#
# with pd.HDFStore(hdf_store) as store:
#     store.put('data/synthetic', pd.DataFrame(generated_data.reshape(-1, n_seq),
#                                              columns=tickers))
#
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 7))
# axes = axes.flatten()
#
# index = list(range(1, 25))
# synthetic = generated_data[np.random.randint(n_windows)]
#
# idx = np.random.randint(len(df) - seq_len)
# real = df.iloc[idx: idx + seq_len]
#
# for j, ticker in enumerate(tickers):
#     (pd.DataFrame({'Real': real.iloc[:, j].values,
#                    'Synthetic': synthetic[:, j]})
#      .plot(ax=axes[j],
#            title=ticker,
#            secondary_y='Synthetic', style=['-', '--'],
#            lw=1))
# sns.despine()
# fig.tight_layout()
