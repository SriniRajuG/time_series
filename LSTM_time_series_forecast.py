import numpy as np
import os
import sys
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping
import tkinter
from matplotlib import pyplot as plt


def create_plot(data_path, data_name, *args):
    fig = plt.figure()
    fig.add_subplot(111)
    for data in args:
        plt.plot(data)
    plt.title(data_name)
    f_name = data_name + ".png"
    f_path = os.path.join(data_path, f_name)
    plt.savefig(f_path)
    plt.close()


def get_sine_data(n_samples):
    # https://stackoverflow.com/questions/22566692/python-how-to-plot-graph-sine-wave
    amplitude = 1
    freq = 15  # no. of oscillations
    # Since amplitude is 1, no need to normalize data.
    x = np.arange(n_samples)
    base1 = 2 * np.pi * freq * x / n_samples
    sine_1 = amplitude * np.sin(base1)
    sine_1 = sine_1.reshape((n_samples, 1))
    base2 = 0.8 * base1
    sine_2 = (np.sin(base1) + np.sin(base2)) / 2
    sine_2 = sine_2.reshape((n_samples, 1))
    sine_data = np.concatenate((sine_1, sine_2), axis=1)
    # sin_1 has a shape (<n_samples>, 1)
    # sin_2 has a shape (<n_samples>, 1)
    # sine_data has a shape (<n_samples>, 2)
    return sine_data


def test_train_split(total_data, shift):
    n_rows = total_data.shape[0]
    n_cols = total_data.shape[1]
    n_rows_train = int(0.8 * n_rows)  # 80% training
    n_rows_remain = n_rows - n_rows_train
    # <n_rows_remain> will be less than the no. of rows in test data
    if shift <= n_rows_remain:
        shape = (1, -1, n_cols)
        x_train = total_data[: n_rows_train, :].reshape(shape)
        y_train = total_data[shift: n_rows_train + shift, :].reshape(shape)
        x_test = total_data[n_rows_train: n_rows - shift, :].reshape(shape)
        y_test = total_data[n_rows_train + shift:, :].reshape(shape)
        # 1 sequence per batch, n_batches = 1
        return x_train, y_train, x_test, y_test


def make_models(n_features):
    # model for training
    model_train = Sequential()
    model_train.add(LSTM(units=100, return_sequences=True, input_shape=(None, n_features)))
    # input_shape=(None, 2)
    # None: Sequence of any length can be given as input. 2: two columns / features in input data
    model_train.add(LSTM(units=70, return_sequences=True))
    model_train.add(LSTM(units=2, return_sequences=True))
    model_train.add(Lambda(lambda x: x * 1.3))  # consider using a fully connected layer?

    # Model for predictions - similar to training model, but uses `return_sequences=False` and `stateful=True`
    model_pred = Sequential()
    model_pred.add(LSTM(units=100, return_sequences=True, stateful=True,
                        batch_input_shape=(1, None, n_features)))
    model_pred.add(LSTM(units=70, return_sequences=True, stateful=True))
    model_pred.add(LSTM(units=2, return_sequences=False, stateful=True))
    model_pred.add(Lambda(lambda x: x * 1.3))
    return model_train, model_pred


def main():
    root_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(root_path, 'data')
    model_f_path = os.path.join(root_path, 'lstm_model.h5')
    n_rows_total = 1200
    shift = 1
    # shift should be >= 1. If shift is k, model will take time-series value at t and predict the value at t+k
    total_data = get_sine_data(n_rows_total)
    n_cols = total_data.shape[1]  # n_cols is the no. of dimensions of the time series
    # create_plot(data_path=data_path, data_name='inp_sin_data', total_data)
    create_plot(data_path, 'inp_sin_data', total_data)
    x_train, y_train, x_test, y_test = test_train_split(total_data, shift)
    # sequence_length is equal to n_elements in each dataset
    # samples in batch = 1, n_batches = 1
    # synced many-to-many RNN in http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    n_rows_train = x_train.shape[1]  # shape of x_train is (1, <n_rows_train>, <n_cols>)
    n_rows_test = x_test.shape[1]
    n_rows_not_train = n_rows_total - n_rows_train
    # n_rows_not_train is greater than n_rows_test due to non-zero value of shift. Look into the test_train_split() for
    # details

    model_train, model_pred = make_models(n_cols)
    # this callback interrupts training when loss stops decreasing after 10 consecutive epochs.
    stop = EarlyStopping(monitor='loss', min_delta=0.000000000001, patience=30)
    model_train.compile(loss='mse', optimizer=Adam(lr=0.0001))
    model_train.fit(x_train, y_train, epochs=1000, callbacks=[stop], verbose=2)
    model_train.save(model_f_path)
    # train indefinitely until loss stops decreasing
    print('\n\n')
    model_pred.set_weights(model_train.get_weights())

    # making predictions from x_test:
    model_pred.reset_states()
    model_pred.predict(x_train)  # to set the states
    y_pred_from_input = list()
    for row_idx in range(n_rows_test):
        y_pred_from_input.append(model_pred.predict(x_test[:, row_idx: row_idx + 1, :]))
        # passing one row at a time, for predictions
        # x_test[:, row_idx: row_idx + 1, :] has the shape (1, 1, 2) and x_test[:, row_idx, :] has the shape (1, 2)
    y_pred_from_input = np.asarray(y_pred_from_input).reshape(1, n_rows_test, n_cols)

    # making predictions from predictions
    new_array = np.empty((1, n_rows_not_train, n_cols))
    new_array[:, :shift, :] = x_test[:, : shift, :]
    # First <shift> elements of x_test are the first <shift> elements of new_array
    model_pred.reset_states()
    model_pred.predict(x_train)  # to set the states
    for row_idx in range(n_rows_test):
        new_array[:, row_idx + shift] = model_pred.predict(new_array[:, row_idx: row_idx + 1, :]).reshape(1, 1, n_cols)
        # Values at <row_idx> of new_array are used for prediction and predictions are again placed in new_array at the
        # index <row_idx> + shift
    y_pred_from_pred = new_array[:, shift:, :]
    # Remove the first <shift> elements from new_array because they are not predictions

    # Plotting for comparing results
    create_plot(data_path, 'self_prediction_col1', y_test[0, :, 0], y_pred_from_pred[0, :, 0])
    create_plot(data_path, 'self_prediction_col2', y_test[0, :, 1], y_pred_from_pred[0, :, 1])
    create_plot(data_path, 'true_data_prediction_col1', y_test[0, :, 0], y_pred_from_input[0, :, 0])
    create_plot(data_path, 'true_data_prediction_col2', y_test[0, :, 1], y_pred_from_input[0, :, 1])


if __name__ == '__main__':
    main()
