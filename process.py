import numpy as np
from sklearn.preprocessing import StandardScaler
from tflearn.data_utils import load_csv
from keras.utils.np_utils import to_categorical as to_cat
from sklearn.externals import joblib

# Process data
def process_data(data, ignore_cols):
    for id in sorted(ignore_cols, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)


ignore = [1, 6]

# Load dataset
data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)
data = process_data(data, ignore)


# Scale the data
scaler = StandardScaler()
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_data = scaler.fit_transform(data)
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)


X = scaled_data[:, 0:6]
y = labels[:, 0]
y = to_cat(y, 2)


# Movie Info for testing
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]


def process(ls, ignore_cols):
    for id in sorted(ignore_cols, reverse=True):
        [ls.pop(id)]
    for i in range(6):
        ls[1] = 1 if ls[1] == 'Female' else 0
    _input = np.reshape(ls, (1, 6))
    _input = scaler.transform(_input)
    return _input


Jack = process(dicaprio, ignore)
Rose = process(winslet, ignore)
