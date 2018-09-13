import tensorflow as tf
# Read MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#####################################################
# 1. Define the layers
#####################################################

#####################################################
# Define a simple convolution layer
# Tensor vector size=[batch, height, width, channels]
# If the input tensor has 4 dimensions:  [batch, height, width, channels]
# stride_vector=[batch_stride, height_stride, width_stride, channels_stride]
# k_size- size of the filter, we assume square size: [k_size X k_size]
# paddingType: 'Same'-size_out=size_in, 'Valid'- size_out=size_in-k_size+1
#####################################################
def conv_layer(input,channels_in,channels_out,k_size,paddingType,stride_vector,name="conv"):
    with tf.name_scope(name):
        W=tf.Variable(tf.truncated_normal([k_size,k_size,channels_in,channels_out],stddev=0.1),name="W")# weights
        b=tf.Variable(tf.constant(0.1,shape=[channels_out]),name="B") # bias
        conv_result=tf.nn.conv2d(input, W, strides=stride_vector, padding=paddingType)
        act=tf.nn.relu(conv_result+b)
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        return act

#####################################################
# Define a fully connected layer
#####################################################
def fully_conn_layer(input,channels_in,channels_out,name="fully_conn"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([channels_in, channels_out],stddev=0.1),name="W")
        b = tf.Variable(tf.constant(0.1,shape=[channels_out]),name="B")
        matmul_result = tf.matmul(input, W)
        act = tf.nn.relu(matmul_result + b)
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        return act

#####################################################
# Define MaxPooling layer
# stride_vector=[batch_stride, height_stride, width_stride, channels_stride]
# k_size- size of the filter, we assume square size: [k_size x k_size]
# paddingType: 'Same'-size_out=size_in, 'Valid'- size_out=size_in-k_size+1
#####################################################
def max_pool_layer(input,k_size,stride_vector, paddingType,name="maxpool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(input,ksize=k_size,strides=stride_vector, padding=paddingType)

#####################################################
# 2. Setup Feed Forward
#####################################################

#####################################################
# Define input placeholders
#####################################################
x = tf.placeholder("float", [None, 784],name="X") # images
data_labels_y = tf.placeholder("float", [None,10],name="labels")   # labels

#####################################################
# Reshape the data
# Setting -1 for one of the dimension means,
# "set the value for the first dimension so that the total number of elements in the tensor is unchanged".
#  In our case the -1 will be equal to the batch_size
#####################################################
x_image = tf.reshape(x, [-1,28,28,1])
tf.summary.image('input', x_image,3)

#####################################################
# Create the network
#####################################################

### 1 Hidden layer ##################################
# 1 Convolution layer
# Size_in=[28,28,1]
# Size_out=[28,28,32]

conv1_channels_in=1
conv1_channels_out=32
conv1_filter_size=5
conv1_paddingType='SAME'
conv1_stride_vector=[1,1,1,1]
conv1_result=conv_layer(x_image,conv1_channels_in,conv1_channels_out,conv1_filter_size,conv1_paddingType,conv1_stride_vector,"conv1")

# MaxPool Layer
# Size_in=[28,28,32]
# Size_out=[14,14,32]
MaxPool1_size=[1,2,2,1]
MaxPool1_stride_vector=[1,2,2,1]
MaxPool1_paddingType='SAME'
Hidden_layer_1_result=max_pool_layer(conv1_result,MaxPool1_size,MaxPool1_stride_vector, MaxPool1_paddingType,"MaxPool1")

### 2 Hidden layer ##################################

# 2 Convolution layer
# Size_in=[14,14,32]
# Size_out=[14,14,64]
conv2_channels_in=conv1_channels_out
conv2_channels_out=64
conv2_filter_size=5
conv2_paddingType='SAME'
conv2_stride_vector=[1,1,1,1]
conv2_result=conv_layer(Hidden_layer_1_result,conv2_channels_in,conv2_channels_out,conv2_filter_size,conv2_paddingType,conv2_stride_vector,"conv2")

# MaxPool Layer
# Size_in=[14,14,64]
# Size_out=[7,7,64]
MaxPool2_size=[1,2,2,1]
MaxPool2_stride_vector=[1,2,2,1]
MaxPool2_paddingType='SAME'
Hidden_layer_2_result=max_pool_layer(conv2_result,MaxPool2_size,MaxPool2_stride_vector, MaxPool2_paddingType,"MaxPool2")
#####################################################

### 1 Fully connected layer ##################################
# Size_in=[7,7,64]
# Size_out=1024
fullconn1_channels_in=7*7*conv2_channels_out
fullconn1_channels_out=1024
# Flatten the output of 2 convolution layer
Hidden_layer_2_result_flat = tf.reshape(Hidden_layer_2_result, [-1, fullconn1_channels_in])
fully_conn_layer_1_result=fully_conn_layer(Hidden_layer_2_result_flat,fullconn1_channels_in,fullconn1_channels_out,"fc1")
#####################################################

### 2 Fully connected layer ##################################
# Size_in=1024
# Size_out=10
fullconn2_channels_in=fullconn1_channels_out
fullconn2_channels_out=10
fully_conn_layer_2_result=fully_conn_layer(fully_conn_layer_1_result,fullconn2_channels_in,fullconn2_channels_out,"fc2")
#####################################################

#####################################################
# 3. Loss & Training setup
#####################################################

# Cross entropy definition
with tf.name_scope("xentropyScope"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=data_labels_y, logits=fully_conn_layer_2_result))
    tf.summary.scalar('cross_entropy', cross_entropy)

# GDS optimizer definition
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Compute the accuracy
# argmax returns the index with the largest value across axis of a tensor
with tf.name_scope("accuracyScope"):
    correct_prediction = tf.equal(tf.argmax(fully_conn_layer_2_result,1), tf.argmax(data_labels_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
#####################################################

#####################################################
# 4. Train the model
#####################################################

sess = tf.Session()
# Initialize all the variables
sess.run(tf.global_variables_initializer())

batch_size=50
numIter=100

###################### TensorBoard ##############################
#################################################################
# Save data for TensorBoard
merged_summary=tf.summary.merge_all()
writter=tf.summary.FileWriter("/output/TensorBoardTest4")
writter.add_graph(sess.graph)

for i in range(numIter):
  # Choose the batch
  batch = mnist.train.next_batch(batch_size)
  # Ocasionaly report accuracy
  if i%10 == 0:
    train_accuracy = sess.run(accuracy,feed_dict={x:batch[0], data_labels_y: batch[1]})
    print("step %d, training accuracy %g" %(i, train_accuracy))

  if i % 2 == 0:
    s=sess.run(merged_summary,feed_dict={x:batch[0], data_labels_y: batch[1]})
    writter.add_summary(s,i)

  # Run the training step
  sess.run(train_step, feed_dict={x: batch[0], data_labels_y: batch[1]})




