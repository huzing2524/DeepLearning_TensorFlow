import tensorflow as tf


v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "C:/Users/huzing2524/Desktop/Projects/machine_learning/05_chapter/Save_and_Restore/model.ckpt")
    # saver.save(sess, "path/to/model/model.ckpt")