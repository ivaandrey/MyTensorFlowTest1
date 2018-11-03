import tensorflow as tf
# Read MNIST data
from DataSetOperations import read_data_sets


##################################################
# 0. Read Data Set
TrainDataSetDir='C:\\Andrey\\DeepLearning\\TensorF\\PointTargetProject\\MydataDir\\Train\\'
TestDataSetDir='C:\\Andrey\\DeepLearning\\TensorF\\PointTargetProject\\MydataDir\\Test\\'
NormalizeImages=True
DataSetToCNN = read_data_sets(TrainDataSetDir,TestDataSetDir,NormalizeImages, one_hot=True)
##################################################

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
def conv_layer(input,channels_in,channels_out,k_size,paddingType,stride_vector,perform_batch_normalization,perform_dropout,name_conv_layer):
    with tf.variable_scope(name_conv_layer):

        ### Weights
        ## weigths initializatiion
        initializer = tf.contrib.layers.xavier_initializer()
        w_shape=[k_size,k_size,channels_in,channels_out]
        W = tf.Variable(initializer(w_shape),name=name_conv_layer+"_W")
        #W=tf.Variable(tf.truncated_normal([k_size,k_size,channels_in,channels_out],stddev=0.1),name="W")# weights

        ## Bias
        ## Biases initialization
        b=tf.Variable(tf.constant(0.1,shape=[channels_out]),name=name_conv_layer+"_B") # bias

        ## DropOut Implementation
        if perform_dropout:
            input = tf.nn.dropout(input, keep_prob=drop_out_keep_prob)

        ## Convolution operation
        conv_result=tf.nn.conv2d(input, W, strides=stride_vector, padding=paddingType)

        if perform_batch_normalization:
        ## Batch Normalization
            conv_result = batch_norm_wrapper(conv_result, is_training, bn_decay_value)


        ## Relu activation
        act=tf.nn.relu(conv_result+b)

        ## Save to histogram in TensorBoard
        tf.summary.histogram(name_conv_layer+'_weights', W)
        tf.summary.histogram(name_conv_layer+'_biases', b)
        tf.summary.histogram(name_conv_layer+'_activations', act)
        tf.summary.histogram(name_conv_layer+'__input_Mult_W', conv_result)

        return act

#####################################################
# Define a fully connected layer
#####################################################
def fully_conn_layer(input,channels_in,channels_out,perform_dropout,name_fully_conn_layer):
    with tf.variable_scope(name_fully_conn_layer, reuse=tf.AUTO_REUSE):

        ### Weights
        ## weigths initializatiion
        initializer = tf.contrib.layers.xavier_initializer()
        w_shape = [channels_in, channels_out]
        W = tf.Variable(initializer(w_shape), name=name_fully_conn_layer + "_W")
      #  W = tf.Variable(tf.truncated_normal([channels_in, channels_out],stddev=0.1),name="W")

        ## Bias
        ## Biases initialization
        b = tf.Variable(tf.constant(0.1,shape=[channels_out]),name=name_fully_conn_layer+"_B")

        ## DropOut Implementation
        if perform_dropout:
            input = tf.nn.dropout(input, keep_prob=drop_out_keep_prob)

        ## Multiplication with weights operation
        matmul_result = tf.matmul(input, W)

        ## Relu activation
        act = tf.nn.relu(matmul_result + b)

        ## Save to histogram in TensorBoard
        tf.summary.histogram(name_fully_conn_layer+'_weights', W)
        tf.summary.histogram(name_fully_conn_layer+'_biases', b)
        tf.summary.histogram(name_fully_conn_layer+'_activations', act)
        tf.summary.histogram(name_fully_conn_layer+'_input_Mult_W',matmul_result)
        return act

#####################################################
# Define Batch Normalization procedure
#####################################################

# this is a simpler version of Tensorflow's 'official' version. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batch_norm_wrapper(inputs, is_training, decay):

    epsilon = 0.00000001
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    batch_mean = tf.cond(is_training, lambda: tf.nn.moments(inputs, [0,1,2])[0],lambda: tf.ones(inputs.get_shape()[-1]) * pop_mean)
    batch_var = tf.cond(is_training, lambda: tf.nn.moments(inputs, [0,1,2])[1], lambda: tf.ones(inputs.get_shape()[-1]) * pop_var)
    train_mean = tf.cond(is_training, lambda: tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay)),
                         lambda: tf.zeros(1))
    train_var = tf.cond(is_training, lambda: tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay)),
                        lambda: tf.zeros(1))

    with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)

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
data_labels_y = tf.placeholder("float", [None,2],name="labels")   # labels

#####################################################
# Define DropOut probability
#####################################################
drop_out_keep_prob = tf.placeholder(tf.float32, name='drop_out_keep_prob')
keep_prob_value=0.8 # dropout keep probability value
#####################################################

#####################################################
# Define BatchNormalization params
#####################################################
is_training = tf.placeholder(tf.bool, name='is_training')
bn_decay_value=0.999

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

# conv1 (5x5x32)
# conv2 (5x5x64)
# fcc1  256
# fcc2  2

### 1 Hidden layer ##################################
# 1 Convolution layer
# Size_in=[28,28,1]
# Size_out=[28,28,32]

conv1_channels_in=1
conv1_channels_out=32
conv1_filter_size=5
conv1_paddingType='SAME'
conv1_stride_vector=[1,1,1,1]
conv1_perform_batch_norm=True
conv1_perform_dropout=False
conv1_name="conv1"
conv1_result=conv_layer(x_image,conv1_channels_in,conv1_channels_out,
                        conv1_filter_size,conv1_paddingType,conv1_stride_vector,
                        conv1_perform_batch_norm,conv1_perform_dropout,conv1_name)

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
conv2_perform_batch_norm=True
conv2_perform_dropout=False
conv2_name="conv2"
conv2_result=conv_layer(Hidden_layer_1_result,conv2_channels_in,conv2_channels_out,
                        conv2_filter_size,conv2_paddingType,conv2_stride_vector,
                        conv2_perform_batch_norm,conv2_perform_dropout,conv2_name)

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
#fullconn1_channels_out=1024
fullconn1_channels_out=256
fullconn1_perform_dropout=True
fullconn1_name="fc1"
# Flatten the output of 2 convolution layer
Hidden_layer_2_result_flat = tf.reshape(Hidden_layer_2_result, [-1, fullconn1_channels_in])
fully_conn_layer_1_result=fully_conn_layer(Hidden_layer_2_result_flat,fullconn1_channels_in,fullconn1_channels_out,
                                           fullconn1_perform_dropout,fullconn1_name)
#####################################################

### 2 Fully connected layer ##################################
# Size_in=1024
# Size_out=2
fullconn2_channels_in=fullconn1_channels_out
fullconn2_channels_out=2
fullconn2_perform_dropout=False
fullconn2_name="fc2"
fully_conn_layer_2_result=fully_conn_layer(fully_conn_layer_1_result,fullconn2_channels_in,fullconn2_channels_out,
                                           fullconn2_perform_dropout,fullconn2_name)
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
if __name__ == '__main__':
    sess = tf.Session()
    # Initialize all the variables
    sess.run(tf.global_variables_initializer())

    # For model saving
    saver = tf.train.Saver()

    batch_size=64
    numIter=30#300

    ###################### TensorBoard ##############################
    #################################################################
    # Save data for TensorBoard
    merged_summary=tf.summary.merge_all()
    writter=tf.summary.FileWriter("./output/TensorBoardTest4")
    writter.add_graph(sess.graph)

    for i in range(numIter):
      # Choose the batch
      batch = DataSetToCNN.train.next_batch(batch_size)

      # Run the training step
      train_step_data = sess.run(train_step,feed_dict={x: batch[0], data_labels_y: batch[1], drop_out_keep_prob: keep_prob_value, is_training: True})

      # Ocasionaly report accuracy
      if i%10 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x:batch[0], data_labels_y: batch[1], drop_out_keep_prob : 1.0, is_training: False})
        correct_pred = sess.run(correct_prediction, feed_dict={x: batch[0], data_labels_y: batch[1], drop_out_keep_prob : 1.0, is_training: False})
        cross_entropy_data = sess.run(cross_entropy, feed_dict={x: batch[0], data_labels_y: batch[1], drop_out_keep_prob : 1.0, is_training: False})

        s = sess.run(merged_summary, feed_dict={x: batch[0], data_labels_y: batch[1], drop_out_keep_prob: 1.0, is_training: False})
        writter.add_summary(s, i)

        print("step %d, training accuracy %g, loss %g" %(i, train_accuracy,cross_entropy_data))



    # Check accuracy on test data
    test_accuracy = sess.run(accuracy,feed_dict={x:DataSetToCNN.test.images, data_labels_y: DataSetToCNN.test.labels, drop_out_keep_prob : 1.0, is_training: False})
    print("test accuracy %g" %(test_accuracy))
    # Save the model
    modelName='./output/SavedModel/my_modelXavier_test_accuracy_' + str(test_accuracy)
    saver.save(sess, modelName)



