#-----------------------------------------------------
# The following is a sample starter code for Project 2,
#  however it does not meet the specifications, and also uses
#  functions that you are not allowed to use.
#
# Refer to project description for specifications on how to
#  implement the project
# Owner: Vinayak Salavi
#-----------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.layers import RepeatVector
from keras import Model
# from keras import optimizers
from keras import metrics
from keras import activations
from keras.layers import RepeatVector
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os


(x_train, label_train), (x_test, label_test) = tf.keras.datasets.mnist.load_data()

# divide data in to  4000 training, 1000 validation, and 1000 test
# images.
training_image = x_train[0:4000]
training_label = label_train[0:4000]
validation_image = x_train[4000:5000]
validation_label = label_train[4000:5000]
testing_image = x_test[0:1000]
testing_label = label_test[0:1000]


# #-----------
# # Save an image to a file
# #-----------

num_train = training_image.shape[0]
num_test  = testing_image.shape[0]
num_val = validation_image.shape[0]

# One-hot encode the training
y_train = np.zeros([num_train, 10])
for i in range(num_train):
    y_train[i, label_train[i]] = 1

# One-hot encode the validation
y_val = np.zeros([num_val,10])
for i in range(num_val):
    y_val[i,validation_label[i]] = 1

# One-hot encode the testing
y_test  = np.zeros([num_test, 10])
for i in range(num_test):
    y_test[i, label_test[i]] = 1


#  create output directory if not present
output_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")

isExist = os.path.exists(output_dir_path)
if isExist!= True:
    os.mkdir(output_dir_path)

for i in range(5):
    print("output path:  ", output_dir_path + "\\test" + str(i) + ".png")
    cv2.imwrite(output_dir_path + "\\test" + str(i) + ".png" , testing_image[i])
    cv2.imwrite(output_dir_path + "\\train" + str(i) + ".png", training_image[i])
    cv2.imwrite(output_dir_path + "\\valid" + str(i) + ".png", validation_image[i])


#--------------------------
# Create the model
#--------------------------

#----------
# Construct the model using layers
#
# NOTE:  For Proj2 you CANNOT use Sequential
#  you must create the layers one at a time as variables
#----------

inputs = keras.Input(shape= (784,))
repeat_vector = layers.RepeatVector(2)(inputs)
reshape_vector = layers.Reshape((28,28,2),input_shape=(784,))(repeat_vector)

#layer 1
conv2d_1 = layers.Conv2D(2,(3,3),activation='relu',padding='same')(reshape_vector)
conv2d_2 = layers.Conv2D(2,(3,3),activation='relu',padding='same')(conv2d_1)
concat_1 = layers.Concatenate()([reshape_vector,conv2d_2])
max_pooling2d = layers.MaxPooling2D((2,2),2)(concat_1)

#layer 2
conv2d_3 = layers.Conv2D(4,(3,3),activation='relu',padding='same')(max_pooling2d)
conv2d_4 = layers.Conv2D(4,(3,3),activation='relu',padding='same')(conv2d_3)
concat_2 = layers.Concatenate()([max_pooling2d,conv2d_4])
max_pooling2d_1 = layers.MaxPooling2D((2,2),2)(concat_2)

#layer 3
conv2d_5 = layers.Conv2D(8,(3,3),activation='relu',padding='same')(max_pooling2d_1)
conv2d_6 = layers.Conv2D(8,(3,3),activation='relu',padding='same')(conv2d_5)
concat_3 = layers.Concatenate()([max_pooling2d_1,conv2d_6])
max_pooling2d_2 = layers.MaxPooling2D((2,2),2)(concat_3)

#layer 4
conv2d_7 = layers.Conv2D(16,(3,3),activation='relu',padding='same')(max_pooling2d_2)
conv2d_8 = layers.Conv2D(16,(3,3),activation='relu',padding='same')(conv2d_7)
concat_4 = layers.Concatenate()([max_pooling2d_2,conv2d_8])
max_pooling2d_3 = layers.MaxPooling2D((2,2),2)(concat_4)

#layer 5
conv2d_9 = layers.Conv2D(32,(3,3),activation='relu',padding='same')(max_pooling2d_3)
conv2d_10 = layers.Conv2D(32,(3,3),activation='relu',padding='same')(conv2d_9)
concat_5 = layers.Concatenate()([max_pooling2d_3,conv2d_10])

flatten = layers.Flatten()(concat_5)
# I observeed that if relu is used as activation for the first dense layer
# accuracy of the model drops to very low value  after certain epoch hence
# used softmax
dense_layer_1 = layers.Dense(10,activation='softmax')(flatten)
dense_layer_2 = layers.Dense(10,activation='softmax')(dense_layer_1)
model = Model(inputs = inputs,outputs = dense_layer_2)
model.summary()

#----------
# Create the loss function
#
# NOTE:  For Proj2 you CANNOT use a pre-made keras loss function
#  you must create the loss function as additional layers
#----------
# loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function
def loss_fn(y_true, y_pred):
    log_value = tf.cast(tf.math.log(y_pred),dtype = tf.float64)
    product_var = tf.math.multiply(y_true,log_value)
    loss = -1 * (K.mean(product_var))
    return loss

#----------
# Compile the model
#
# NOTE:  For Proj2 you CANNOT just compile the model,
#  you must create an optimizer variable as listed in
#  keras.optimizers, and give the model's training weights
#  to the optimizer.
#----------
# learning rates tried 5e-3,5e-4,1e-3,1e-4
optimizer = keras.optimizers.Adam(learning_rate = 8e-3)

#----------
# Fit the model
#
# NOTE:  For Proj2 you CANNOT call model.fit
#  you must create a custom training loop with pseudocode
#
# batch_size = 100
# num_batches = num_train / batch_size
#
# for e in range(num_epochs):
#     for b in range(num_batches):
#         # perform gradient update over minibatch using "gradient tape"
#----------


# variable declaration for custom training loop
epoch_nums = 150
batch_size = 100

# arrays to store losses and accuracy over the iterations
training_loss_arr = []
training_accuracy_arr = []
validation_loss_arr = []
validation_accuracy_arr = []
testing_accuracy_arr = []

# define accuracy metrics
train_accuracy_metric = metrics.CategoricalAccuracy()
validation_accuracy_metric = metrics.CategoricalAccuracy()
train_accuracy_metric = metrics.CategoricalAccuracy()

# Prepare the training dataset
training_image = training_image / 255.0
training_image = np.reshape(training_image,(-1,784))
train_dataset = tf.data.Dataset.from_tensor_slices((training_image,y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

validation_image = validation_image / 255.0
validation_image = np.reshape(validation_image,(-1,784))

testing_image = testing_image / 255.0
testing_image = np.reshape(testing_image,(-1,784))

# custom training loop 
for epoch in range(epoch_nums):
    print("\nStarting epoch %d" % (epoch,))
    total_Loss = 0
    # iterate over the batches of the dataset
    for step, (xBatch_train, yBatch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(xBatch_train, training=True)
            train_accuracy_metric.update_state(yBatch_train,logits)
            current_loss = loss_fn(yBatch_train,logits)
        grads = tape.gradient(current_loss,model.trainable_weights)
        optimizer.apply_gradients(zip(grads,model.trainable_weights))
        current_loss = tf.math.reduce_sum(current_loss).numpy()
        total_Loss = (total_Loss + current_loss)/num_train
    training_loss_arr.append(total_Loss)
    current_accuracy = train_accuracy_metric.result()
    print(f"Accuracy : {current_accuracy}\t loss : {total_Loss}")
    training_accuracy_arr.append(current_accuracy.numpy())

    train_accuracy_metric.reset_states()
    validation_predictions = model(validation_image, training=False)
    current_validation_loss = (tf.reduce_sum(loss_fn(y_val, validation_predictions)).numpy())/num_val
    validation_loss_arr.append(current_validation_loss)
    validation_accuracy_metric.update_state(y_val, validation_predictions)
    current_validation_accuracy = validation_accuracy_metric.result()
    print(f"\nvalidation accuracy : {current_validation_accuracy}\t validation loss : {current_validation_loss}")
    validation_accuracy_arr.append(current_validation_accuracy.numpy())
    validation_accuracy_metric.reset_states()

# tesing loop
testing_predictions = model(testing_image, training=False)
train_accuracy_metric.update_state(y_test, testing_predictions)
testing_accuracy_arr = train_accuracy_metric.result().numpy()
train_accuracy_metric.reset_states()
print(f" Testing set Accuracy : {testing_accuracy_arr}")
testAccuracy_arr = np.full((epoch_nums), testing_accuracy_arr)

# plot graphs
x_axis = range(epoch_nums)
plt.plot(x_axis, training_loss_arr, color="red",label="train")
plt.plot(x_axis, validation_loss_arr, color="black",label="valid")
plt.title('loss')
plt.legend()
plt.savefig(output_dir_path + "\\loss.png")
plt.close()

plt.title('accuracy')
plt.plot(x_axis, training_accuracy_arr, color="red",label="train")
plt.plot(x_axis, validation_accuracy_arr, color="black", label="valid")
plt.plot(x_axis, testAccuracy_arr, color="blue", label="test")
plt.legend()
plt.savefig(output_dir_path + "\\accuracy.png")


# Reference:
# https://keras.io/guides/writing_a_training_loop_from_scratch/
# https://towardsdatascience.com/how-to-create-a-custom-loss-function-keras-3a89156ec69b


