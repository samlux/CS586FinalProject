from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
plt.style.use("ggplot")



tf.reset_default_graph()
n_hidden = 64
n_classes = 1
n_features = 1

X_ = tf.placeholder(tf.float32, shape = [None, n_features])
y_ = tf.placeholder(tf.float32, shape = [None, n_classes])

with tf.variable_scope("parameters"):
    w1 = tf.Variable(tf.random_uniform([n_features, n_hidden]))
    b1 = tf.Variable(tf.random_uniform([n_hidden]))
    w2 = tf.Variable(tf.random_uniform([n_hidden, n_hidden]))
    b2 = tf.Variable(tf.random_uniform([n_hidden]))
    w3 = tf.Variable(tf.random_uniform([n_hidden, n_hidden]))
    b3 = tf.Variable(tf.random_uniform([n_hidden]))
with tf.variable_scope("model"):
    z1 = tf.matmul(X_, w1) + b1
    fc1 = tf.nn.tanh(z1)
    z2 = tf.matmul(fc1, w2) + b2
    fc2 = tf.nn.tanh(z2)
    z3 = tf.matmul(fc2, w3) + b3

loss = tf.reduce_mean(tf.square(z3-y_))
op = tf.train.AdamOptimizer(1e-2).minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_tasks = []
# load json and create model
json_file = open('Models/NJ_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
NJ_loaded_model = model_from_json(loaded_model_json)
# load weights into new model
NJ_loaded_model.load_weights("Models/NJ_model.h5")
print("Loaded NJ_model from disk")
n_tasks.append(NJ_loaded_model)

# load json and create model
json_file = open('Models/NY_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
NY_loaded_model = model_from_json(loaded_model_json)
# load weights into new model
NY_loaded_model.load_weights("Models/NY_model.h5")
print("Loaded NY_model from disk")
n_tasks.append(NY_loaded_model)

# load json and create model
json_file = open('Models/CA_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
CA_loaded_model = model_from_json(loaded_model_json)
# load weights into new model
CA_loaded_model.load_weights("Models/CA_model.h5")
print("Loaded CA_model from disk")
n_tasks.append(CA_loaded_model)

# load json and create model
json_file = open('Models/TX_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
TX_loaded_model = model_from_json(loaded_model_json)
# load weights into new model
TX_loaded_model.load_weights("Models/TX_model.h5")
print("Loaded TX_model from disk")
n_tasks.append(TX_loaded_model)

# load json and create model
json_file = open('Models/WA_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
WA_loaded_model = model_from_json(loaded_model_json)
# load weights into new model
WA_loaded_model.load_weights("Models/WA_model.h5")
print("Loaded WA_model from disk")
n_tasks.append(WA_loaded_model)

# load json and create model
json_file = open('Models/FL_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
FL_loaded_model = model_from_json(loaded_model_json)
# load weights into new model
FL_loaded_model.load_weights("Models/FL_model.h5")
print("Loaded FL_model from disk")
n_tasks.append(FL_loaded_model)

for model in n_tasks:
    testX = np.array((0.5, 0.75))
    testX = np.reshape(testX, (testX.shape[0], 1, 1))
    print(model.predict(testX))
