import tensorflow as tf

ta = tf.zeros((2,2))

print(ta)

# Throws error
#print(ta.eval())

session = tf.InteractiveSession()
print(ta.eval())
session.close()


'''
Constants, Variables
'''
W1 = tf.zeros((3,3))
W2 = tf.Variable(tf.zeros((2,2)), name='weight')

session = tf.InteractiveSession()
print(W1.eval())
# Throws error
#print(W2.eval())

session.run(tf.global_variables_initializer())
print(W1.eval())
print(W2.eval())
session.close()

'''
Placeholder, feed_dict
'''
input1 = tf.placeholder(tf.float32, 3)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

session = tf.InteractiveSession()
out_val = session.run([output], feed_dict={input1:[1., 2., 3.], input2:[3.]})

print(out_val)
session.close()


'''
Tensorboard#FileWriter
'''
session = tf.InteractiveSession()
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b)

writer = tf.summary.FileWriter('/tmp/graphs', session.graph)

session.run(x)

writer.close()
session.close()

# Connect to: http://imstyles:6006

