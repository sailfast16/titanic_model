from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import tensorflow as tf
from process import X, y

# Lets Tensorflow utilize more of the GPU for calculations
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# Neural Network Hyper Parameters
num_epochs = 20  # Number of training iterations
batch_size = 20  # How many data points to train on at a time


# Allows visualization of training statistics on localhost:6006
callback = TensorBoard(log_dir='logs',
                       histogram_freq=1,
                       write_graph=True,
                       write_images=True,
                       write_grads=True,
                       batch_size=batch_size)


# Construct Neural Network Graph
model = Sequential()
model.add(Dense(70, input_dim=6, activation='relu'))
model.add(Dense(50, activation='softmax'))
model.add(Dense(30, activation='softmax'))
model.add(Dense(2, activation='softmax'))

# Compile the Graph
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit data to the model
model.fit(X, y,
          epochs=num_epochs,
          validation_split=.1,  # fraction of training data to use to check the model performance
          batch_size=batch_size,
          verbose=1,
          callbacks=[callback])

# Evaluate the model's performance
scores = model.evaluate(X, y, verbose=0)
print(scores)

# Save the model so it doesn't have to retrain every time you want to make a prediction
model.save('model_data/model.h5')
print('model saved to disk')




