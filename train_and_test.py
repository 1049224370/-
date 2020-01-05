import glob
import os
import tensorflow as tf
import numpy as np
import time
# import cv2
import pandas as pd
import sys
model_path = './model'



# -----------------read files----------------------
TRAIN_VECTORS_ALL = []
TRAIN_LABELS_ALL = []
NUM_CHECKPOINTS = 100
max_words_scale=56
phrase_ids_train = []
file = open('train_vectors.txt', 'r')

max_word_scale = 0
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        pass  # do something
    # vectors_of_a_sentence_float = []
    for i in lines:
        vectors_of_a_sentence_float = []
        i = i.strip('\n')
        vectors_of_a_sentence_str = i.split(',')
        word_scale = len(vectors_of_a_sentence_str)
        if(word_scale > max_word_scale):
            max_word_scale = word_scale
        for vectors_of_a_word_str in vectors_of_a_sentence_str:
            if(len(vectors_of_a_word_str) == 0):
                continue
            temp = vectors_of_a_word_str.split(' ')
            temp = temp[0:50]
            if(len(temp)==50):
                vectors_of_a_word_float = list(map(float, temp))
                vectors_of_a_word_float = np.asarray(vectors_of_a_word_float)
                vectors_of_a_sentence_float.append(vectors_of_a_word_float)
        TRAIN_VECTORS_ALL.append(vectors_of_a_sentence_float)
file.close()
print("max word scale = ",max_words_scale)
print(len(TRAIN_VECTORS_ALL))
# phrase_ids_train = np.array(phrase_ids_train)
# print(phrase_ids_train)
file = open('train_labels.txt', 'r')
# max_word_scale = 0
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        pass  # do something
    for i in lines:
        TRAIN_LABELS_ALL.append(int(float(i.strip('\n'))/0.2))
print(len(TRAIN_LABELS_ALL))
file.close()

#      ___________split dev______________
DEV_VECTORS_ALL = []
DEV_LABELS_ALL = []

phrase_ids_train = []
file = open('dev_vectors.txt', 'r')

# max_word_scale = 0
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        pass  # do something
    # vectors_of_a_sentence_float = []
    for i in lines:
        vectors_of_a_sentence_float = []
        i = i.strip('\n')
        vectors_of_a_sentence_str = i.split(',')
        # word_scale = len(vectors_of_a_sentence_str)
        # if(word_scale > max_word_scale):
        #     max_word_scale = word_scale
        # counter = 0
        for vectors_of_a_word_str in vectors_of_a_sentence_str:
            if(len(vectors_of_a_word_str) == 0):
                continue
            temp = vectors_of_a_word_str.split(' ')
            temp = temp[0:50]
            if(len(temp)==50):
                vectors_of_a_word_float = list(map(float, temp))
                vectors_of_a_word_float = np.asarray(vectors_of_a_word_float)
                vectors_of_a_sentence_float.append(vectors_of_a_word_float)
        #     counter += 1
        # print(counter)
        # print(len(vectors_of_a_sentence_float))
        DEV_VECTORS_ALL.append(vectors_of_a_sentence_float)
file.close()
# print(max_word_scale)
# print(len(DEV_VECTORS_ALL))
# phrase_ids_train = np.array(phrase_ids_train)
# print(phrase_ids_train)
file = open('dev_labels.txt', 'r')
# max_word_scale = 0
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        pass  # do something
    for i in lines:
        DEV_LABELS_ALL.append(int(float(i.strip('\n'))/0.2))
print(len(DEV_LABELS_ALL))
file.close()

TEST_VECTORS_ALL = []
TEST_LABELS_ALL = []

phrase_ids_train = []
file = open('test_vectors.txt', 'r')

# max_word_scale = 0
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        pass  # do something
    # vectors_of_a_sentence_float = []
    for i in lines:
        vectors_of_a_sentence_float = []
        i = i.strip('\n')
        vectors_of_a_sentence_str = i.split(',')
        # word_scale = len(vectors_of_a_sentence_str)
        # if(word_scale > max_word_scale):
        #     max_word_scale = word_scale
        # counter = 0
        for vectors_of_a_word_str in vectors_of_a_sentence_str:
            if(len(vectors_of_a_word_str) == 0):
                continue
            temp = vectors_of_a_word_str.split(' ')
            temp = temp[0:50]
            if(len(temp)==50):
                vectors_of_a_word_float = list(map(float, temp))
                vectors_of_a_word_float = np.asarray(vectors_of_a_word_float)
                vectors_of_a_sentence_float.append(vectors_of_a_word_float)
        #     counter += 1
        # print(counter)
        # print(len(vectors_of_a_sentence_float))
        TEST_VECTORS_ALL.append(vectors_of_a_sentence_float)
file.close()
# print(max_word_scale)
# print(len(DEV_VECTORS_ALL))
# phrase_ids_train = np.array(phrase_ids_train)
# print(phrase_ids_train)
file = open('test_labels.txt', 'r')
# max_word_scale = 0
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        pass  # do something
    for i in lines:
        TEST_LABELS_ALL.append(int(float(i.strip('\n'))/0.2))
print(len(TEST_LABELS_ALL))
file.close()
# -----------------end read files----------------------

#--------------------data reshape------------------------
NP_TRAIN_VECTORS = np.zeros(shape=(len(TRAIN_VECTORS_ALL),max_words_scale,50),dtype=np.float32)
NP_TRAIN_LABELS = np.asarray(TRAIN_LABELS_ALL)

for i in range(0,len(TRAIN_VECTORS_ALL)):
    # print("words in a sentence=")
    # print(len(TRAIN_VECTORS_ALL[i]))
    for j in range(0,len(TRAIN_VECTORS_ALL[i])-1):
        NP_TRAIN_VECTORS[i][j] = TRAIN_VECTORS_ALL[i][j]

NP_DEV_VECTORS = np.zeros(shape=(len(DEV_VECTORS_ALL),max_words_scale,50),dtype=np.float32)
NP_DEV_LABELS = np.asarray(DEV_LABELS_ALL)


for i in range(0,len(DEV_VECTORS_ALL)):
    for j in range(0,len(DEV_VECTORS_ALL[i])-1):
        NP_DEV_VECTORS[i][j] = DEV_VECTORS_ALL[i][j]

NP_TEST_VECTORS = np.zeros(shape=(len(TEST_VECTORS_ALL),max_words_scale,50),dtype=np.float32)
NP_TEST_LABELS = np.asarray(TEST_LABELS_ALL)


for i in range(0,len(TEST_VECTORS_ALL)):
    for j in range(0,len(TEST_VECTORS_ALL[i])-1):
        NP_TEST_VECTORS[i][j] = TEST_VECTORS_ALL[i][j]
#         zero padding
#--------------------data reshape complete------------------------




# 打乱顺序
num_example = NP_TRAIN_VECTORS.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
train_data = NP_TRAIN_VECTORS[arr]
train_label = NP_TRAIN_LABELS[arr]
n_epoch = 120000
batch_size = 40
# divided datas into train and validation


def yield_batches(inputs=None, targets=None, batch_size=None, shuffle=False):

    if(len(inputs) != len(targets)):
        print("!!!!!!!!!!data not fit!!!!!!!!!!")
        return None
    if shuffle:
        indexs = np.arange(len(inputs))
        np.random.shuffle(indexs)
    else:
        indexs = np.arange(len(inputs))
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        # print("i am angry")
        # print(targets)
        if shuffle:
            # print("a",start_idx,start_idx + batch_size)
            indexs_ = indexs[start_idx: start_idx + batch_size]
        else:

            # print("b")
            indexs_ = indexs[start_idx: start_idx + batch_size]

        out_vectors = []
        out_labels = []
        # print(indexs_)
        out_labels = np.array(targets)[indexs_]
        # print(out_labels)
        # print("index", indexs_)
        for vector in (inputs[indexs_]):
            out_vectors.append(vector)
            out_vectors_ = np.array(out_vectors,dtype=np.float32)
        # if(len(targets[indexs]) == 0):
        #     print("WTFWTF")
        #     print(indexs)
        # print(out_vectors_,out_labels)
        yield out_vectors_, out_labels
# -----------------place holders----------------------

x = tf.placeholder(tf.float32, shape=[None, max_words_scale, 50, 1], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


def inference(input_tensor, train, regularizer):
    print("input x = ", input_tensor)
    with tf.variable_scope('layer_1_conv_0'):
        conv0_weights = tf.get_variable("weight", [5,50, 1, 2],
                                        initializer=tf.truncated_normal_initializer(stddev=0.2))
        conv0_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.0))
        conv0 = tf.nn.conv2d(input_tensor, conv0_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu0 = tf.nn.relu(tf.nn.bias_add(conv0, conv0_biases))
        print("conv0 = ", conv0)
        print("relu0 = ", relu0)

    with tf.variable_scope('layer_1_conv_1'):
        conv1_weights = tf.get_variable("weight", [4,50, 1, 2],
                                        initializer=tf.truncated_normal_initializer(stddev=0.2))
        conv1_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        print("conv1 = ", conv1)
        print("relu1 = ", relu1)

    with tf.variable_scope('layer_1_conv_2'):
        conv2_weights = tf.get_variable("weight", [3, 50, 1, 2],
                                        initializer=tf.truncated_normal_initializer(stddev=0.2))
        conv2_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(input_tensor, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        print("conv2 = ", conv2)
        print("relu2 = ", relu2)

    with tf.variable_scope('layer_1_conv_3'):
        conv3_weights = tf.get_variable("weight", [2, 50, 1, 2],
                                        initializer=tf.truncated_normal_initializer(stddev=0.2))
        conv3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(input_tensor, conv3_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        print("conv3 = ", conv3)
        print("relu3 = ", relu3)

    with tf.name_scope("layer_2_pool_0"):
        pool0 = tf.nn.max_pool(relu0, ksize=[1, max_words_scale-4, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        print("pool0 = ", pool0)
    with tf.name_scope("layer_2_pool_1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, max_words_scale-3, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        print("pool1 = ", pool1)
    with tf.name_scope("layer_2_pool_2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, max_words_scale-2, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        print("pool2 = ", pool2)
    with tf.name_scope("layer_2_pool_3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, max_words_scale-1, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        print("pool3 = ", pool3)

    with tf.name_scope("layer_3_pile_up"):
        pool_6_uni_vector = tf.concat([pool0,pool1,pool2,pool3],2)
        print("pool_6_uni_vector = ", pool_6_uni_vector)
        nodes = 8
        reshaped = tf.reshape(pool_6_uni_vector, [-1, nodes])
        print("reshaped = ",reshaped)

    with tf.variable_scope('layer4-fc'):
        fc_weights = tf.get_variable("weight", [nodes, 5],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc_weights))
        fc_biases = tf.get_variable("bias", [5], initializer=tf.constant_initializer(0.0))
        logit = tf.matmul(reshaped, fc_weights) + fc_biases
        # logit = tf.cast((logit/0.2),dtype=tf.int32)
    print("logit = ", logit)
    return logit



# ---------------------------end of nets---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x, True, regularizer)
# print("a")
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')
# print("b")
# print(logits,y_)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y_,5),name="loss")
print(loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss,name='train_op')
# print("c")
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_,name="correct_prediction")
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accepted")
# print("d")
saver = tf.train.Saver(
    tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1  # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True  # allocate dynamically
sess = tf.Session(config = config)
# print("e")
sess.run(tf.global_variables_initializer())
flag = 0
while(1):
    if (flag==1):
        break
    for epoch in range(n_epoch):
        print("n_epoch = ", epoch)
        start_time = time.time()

        # training

        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in yield_batches(NP_TRAIN_VECTORS, NP_TRAIN_LABELS, batch_size, shuffle=True):
            x_train_a = x_train_a[:,:,:,np.newaxis]
            _, error, accepted = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += error;
            train_acc += accepted;
            n_batch += 1
            if(n_batch%300 == 0):
                print("n_batch = ",n_batch)
        print("   train loss: %f" % (np.sum(train_loss) / n_batch))
        print("   train acc: %f" % (np.sum(train_acc) / n_batch))
        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val, y_val in yield_batches(NP_DEV_VECTORS, NP_DEV_LABELS, batch_size, shuffle=False):
            x_val = x_val[:, :, :, np.newaxis]
            error, accepted = sess.run([loss, acc], feed_dict={x: x_val, y_: y_val})
            val_loss += error;
            val_acc += accepted;
            n_batch += 1
        # this is a MA for loss or acc
        print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
        print("   validation acc: %f" % (np.sum(val_acc) / n_batch))

        if((np.sum(val_acc) / n_batch)>=0.38):
            test_loss, test_acc, n_batch = 0, 0, 0
            for x_val, y_val in yield_batches(NP_TEST_VECTORS, NP_TEST_LABELS, batch_size, shuffle=False):
                x_val = x_val[:, :, :, np.newaxis]
                error, accepted = sess.run([loss, acc], feed_dict={x: x_val, y_: y_val})
                test_loss += error;
                test_acc += accepted;
                n_batch += 1
            # this is a MA for loss or acc
            # print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
            print("   test acc: %f" % (np.sum(test_acc) / n_batch))
            flag=1
            break

        if(n_epoch%2 == 0):
            saver.save(sess, model_path,global_step=n_epoch)


    saver.save(sess, model_path)
saver.save(sess, model_path)
sess.close()
