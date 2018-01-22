import tensorflow as tf
import winsound

frequency = 2500
duration = 60000

traindata = [1, 3, 4, 6, 7, 9, 11, 14, 28, 29, 30, 32, 34, 37, 38] # 1, 3, 4, 6
validatedata = [20, 21, 23, 27]
filename = ['D2N2Sur', 'H2N2A', 'H2N2C', 'H2N2D', 'H2N2S', 'N2A', 'N2C', 'N2D', 'N2H', 'N2S', 'N2Sur', 'S2N2H']
maxframe = 444

# facial landmark point => 68
# read data from file
# f = open('FacialLandmarking/Train1/D2N2Sur.txt', 'r')
videodata = []
for packet in traindata:
    for file in filename:
        f = open('FacialLandmarking/Train'+str(packet)+'/'+file+'.txt', 'r')
        counter = 0
        video = []
        for line in f:
            temp = line.replace('\n', '').replace('[', '').replace(']', '').split(' ')
            temp = temp[:-1]
            video.append(temp)
            counter += 1
        f.close()
        counter = maxframe - counter
        if counter > 0:
            temp = [0]*136
            for i in range(counter):
                video.append(temp)
        videodata.append(video)
# print(video)

videovalidate = []
for val in validatedata:
    for name in filename:
        f = open('FacialLandmarking/Train'+str(val)+'/'+name+'.txt', 'r')
        counter = 0
        video = []
        for line in f:
            temp = line.replace('\n', '').replace('[', '').replace(']', '').split(' ')
            temp = temp[:-1]
            video.append(temp)
            counter += 1
        f.close()
        counter = maxframe - counter
        if counter > 0:
            temp = [0]*136
            for i in range(counter):
                video.append(temp)
        videovalidate.append(video)

# make desired output
desiredoutput = []
temp = [0] * 12
for i in range(len(traindata)):
    for j in range(len(filename)):
        ins = temp
        ins[j] = 1
        desiredoutput.append(ins)

outputvalidate = []
for i in range(len(validatedata)):
    for j in range(len(filename)):
        ins = temp
        ins[j] = 1
        outputvalidate.append(ins)

globalstep = tf.Variable(0, trainable=False)
learningrate = 0.001
lr = tf.maximum(tf.train.exponential_decay(learningrate, globalstep, 100, 0.99, staircase=True), 0.00001)
training_iters = 200000
displaystep = 100

n_input = 136
n_step = maxframe
n_hidden_unit = 600
n_classes = len(filename)

# tf graph input
x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# define weight
weight = {
    'in': tf.Variable(tf.random_normal([n_input, n_hidden_unit]), name='weightin'),
    'out': tf.Variable(tf.random_normal([n_hidden_unit, n_classes]), name='weightout')
}
bias = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unit, ]), name='biasin'),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]), name='biasout')
}


def RNN(x, weight, bias):
    x = tf.reshape(x, [-1, n_input])
    x_in = tf.matmul(x, weight['in']) + bias['in']
    x_in = tf.reshape(x_in, [-1, n_step, n_hidden_unit])
    x_in = tf.nn.dropout(x_in, keep_prob=keep_prob)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unit, state_is_tuple=True)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, time_major=False, dtype=tf.float32)
    result = tf.matmul(final_state[1], weight['out']) + bias['out']
    return result


pred = RNN(x, weight, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost, global_step=globalstep)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# print(len(videodata))
# print(len(desiredoutput))

# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), feed_dict={x: videodata, y: desiredoutput, keep_prob: 1}))
#     # print(sess.run(pred, feed_dict={x: videodata}))
#     # print(sess.run(tf.argmax(desiredoutput)))

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, 'FacialLandmarking/save.ckpt')
    for i in range(training_iters):
        sess.run(train_op, feed_dict={x: videodata, y: desiredoutput, keep_prob: 0.5})
        if i % displaystep == 0:
            temp = sess.run(accuracy, feed_dict={x: videodata, y: desiredoutput, keep_prob: 1})
            print(temp)
            if temp > 0.5:
                break
    save = saver.save(sess, 'FacialLandmarking/save.ckpt')

# with tf.Session() as sess:
#     saver.restore(sess, 'FacialLandmarking/save.ckpt')
#     temp = sess.run(accuracy, feed_dict={x: videovalidate, y: outputvalidate, keep_prob: 1})
#     print(temp)
#     # print(sess.run(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), feed_dict={x: videovalidate, y: outputvalidate, keep_prob: 1}))

winsound.Beep(frequency, duration)
