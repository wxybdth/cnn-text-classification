import  tensorflow as tf
#cnn 看得懂版
filter_size = [2,3] #滤波器的长度分别为2和3
filter_num = 5
embedding = tf.constant([[0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2],[0.3,0.3,0.3,0.3],[0.4,0.4,0.4,0.4]]) #vocab_size 4 embedding_size = 4

#输入长度句子长度为5
input = tf.placeholder(tf.int32, [None, 5])
with tf.name_scope('embedding'):
    W = tf.Variable(embedding, name='embedding')
    embed_vec = tf.nn.embedding_lookup(embedding, input)
    embed_vec_shape = tf.shape(embed_vec) #[2,5,4] [batch_size, seq_len, embedding_size]
    embed_vec_flat = tf.expand_dims(embed_vec, -1)  #由三维变成四维，变成[2,5,4,1]，因为卷积操作输入的为四维
    #分别为[batch_size, hight, width, channels] channels对应图像rgb
pooled_res = []
for filter1 in filter_size:
    with tf.name_scope('conv-maxpooling-%s'%filter1):
        filters = [filter1,4,1,5] #[filter_size(2或3),embedding_size,1,filters(卷积核个数)]
        W = tf.Variable(tf.random_uniform(filters,-1,1)) #[2,4,1,5]or[3,4,1,5]
        b = tf.Variable(tf.constant(0.1,shape=[5])) #卷积核个数
        # 卷积参数，embedding后的结果，卷积核，padding模式可以选择是补0还是不补0
        conv = tf.nn.conv2d(embed_vec_flat, W, padding='VALID',strides=[1,1,1,1])
        conv_shape = tf.shape(conv) #[2,3,1,5] 长度为3,5-3+1=3
        #max_pool ksize表示maxpooling的范围大小[第一维在batch上做,第二维在高度上，第三维宽度上，第四维channel上]
        h_pool = tf.nn.max_pool(conv,ksize=[1,5-filter1+1,1,1],strides = [1,1,1,1],padding='VALID')
        h_pool_shape = tf.shape(h_pool) #[2,1,1,5]
        pooled_res.append(h_pool)
num_filters_total =5 * len(filter_size)
#tf.concat 将每个卷积核的结果进行合并，pooled_res待合并矩阵，3是合并维度 [2,1,1,5]合并上[2,1,1,5]
h_pool_outputs = tf.concat(pooled_res, 3) #[2,1,1,10]
h_pool_outputs_shape = tf.shape(h_pool_outputs)

#将得到的结果reshape，[2,1,1,10]变成[-1,10],10为确定的维度，-1为不知道确定有多少，但是可以计算出
h_pool_outputs_flatten = tf.reshape(h_pool_outputs, [-1, num_filters_total])
h_pool_outputs_flatten_shape = tf.shape(h_pool_outputs_flatten) #[2,10]

with tf.name_scope('drop_out'):
    drop_out = tf.nn.dropout(h_pool_outputs_flatten, 0.5)

with tf.name_scope('prediction'):
    W = tf.Variable(tf.truncated_normal([num_filters_total, 2]))
    b = tf.Variable(tf.constant(0.1,shape=[2]))
    res1 = tf.nn.xw_plus_b(drop_out,W,b) #[2,2] [batch_size, num_class]
    prediction = tf.argmax(res1,axis=1)


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print('variable initalization done!')
    print('embed_vec_size :',sess.run(embed_vec_shape, feed_dict={input:[[0,1,2,3,3],[2,3,0,1,0]]}))

    print('conv size :',sess.run(conv_shape, feed_dict={input: [[0, 1, 2, 3, 3], [2, 3, 0, 1, 0]]}))
    print('h_pool size :',sess.run(h_pool_shape, feed_dict={input: [[0, 1, 2, 3, 3], [2, 3, 0, 1, 0]]}))

    print('h_pool_output :', sess.run(h_pool_outputs_shape, feed_dict={input: [[0, 1, 2, 3, 3], [2, 3, 0, 1, 0]]}))
    print('h_pool_outputs_flatten_shape :',sess.run(h_pool_outputs_flatten_shape, feed_dict={input:[[0,1,2,3,3],[2,3,0,1,0]]}))

    # print(sess.run(res, feed_dict={input:[[0,1,2,3,3],[2,3,0,1,0]]}))
    print('prediction',sess.run(res1,feed_dict={input:[[0,1,2,3,3],[2,3,0,1,0]]}))




