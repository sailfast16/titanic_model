from keras.models import load_model
import tensorflow as tf
from sklearn.externals import joblib
from process import process, ignore


# people data
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]

Jack = process(dicaprio, ignore)
Rose = process(winslet, ignore)

# how to scale the inputs for training
scaler_filename = 'scaler.save'
scaler = joblib.load(scaler_filename)

# lets python use more of my GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# import trained model and predict
model = load_model('model_data/model.h5')
pred = model.predict(Rose)

print( "chance of survival is: ", pred[0][0])



