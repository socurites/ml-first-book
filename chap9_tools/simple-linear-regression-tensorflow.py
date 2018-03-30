import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets


# Load datasets
boston = datasets.load_boston()
boston_slice = [x[5] for x in boston.data]      # RM: average number of rooms per dwelling

data_x = np.array(boston_slice).reshape(-1, 1)
data_y = boston.target.reshape(-1, 1)

print("# data format: #samples X #features")
print(data_x.shape, data_y.shape)


# Define placeholders for inputs
n_sample = data_x.shape[0]
X = tf.placeholder(tf.float32, shape=(n_sample,1), name='X')
y = tf.placeholder(tf.float32, shape=(n_sample,1), name='Y')


# Define model: variables for weights
# model = linear regression: y = wx + b
W = tf.Variable(tf.zeros((1,1)), name='weights')      # (#features, #targets)
b = tf.Variable(tf.zeros((1,1)), name='bias')         # (#targets, #targets)

y_pred = tf.matmul(X,W) + b


# Define loss
loss = tf.reduce_mean(tf.square(y_pred-y))


# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

summary_op = tf.summary.scalar('loss', loss)

'''
Util plot function
'''
def plot_graph(y):
    print(data_x.reshape(-1))

    plt.scatter(data_x.reshape(-1), boston.target.reshape(-1))
    plt.plot(data_x.reshape(-1), y.reshape(-1))
    plt.show()

# Initialize variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('/tmp/graphs', sess.graph)

    plot_graph(sess.run(y_pred, {X: data_x}))


# Train model by running optimizer
    for i in range(100):
        loss_t, summary, _ = sess.run([loss, summary_op, train_op],
                                      feed_dict={X: data_x, y: data_y})

        summary_writer.add_summary(summary, i)

        if i%10 == 0:
            print('loss = % 4.4f' % loss_t.mean())

    plot_graph(sess.run(y_pred, {X: data_x}))