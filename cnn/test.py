import tensorflow as tf

a = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
b = tf.expand_dims(a,-1)
c = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
d = tf.expand_dims(c,-1)
size_a = tf.size(a) #3
shape_a = tf.shape(a)
b = tf.expand_dims(a,-1)
size_b = tf.size(b)
shape_b = tf.shape(b)
e = tf.concat([b,d],axis=3)
shape_e = tf.shape(e)
with tf.Session() as sess:
    print('a:' ,sess.run(a))
    print('size_a :',sess.run(size_a))
    print('shape_a', sess.run(shape_a))
    print('b: ',sess.run(b))
    print('size_b :', sess.run(size_b))
    print('shape_b : ', sess.run(shape_b))
    print('c_concat_d',sess.run(e))
    print(('shape c_concat_d',sess.run(shape_e)))




