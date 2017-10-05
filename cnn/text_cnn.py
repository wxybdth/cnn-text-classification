import tensorflow as tf

class TextCNN():
    def __init__(
            self,sequence_len, num_class, num_vocab, embedding_size, filter_size_list, num_filters,l2_reg_lambda):
        self.input = tf.placeholder(tf.int32, [None,sequence_len])
        self.output = tf.placeholder(tf.float32,[None, num_class])
        self.hold_keep_hold = tf.placeholder(tf.float32)
        l2_loss = tf.constant(0.0)

        #embedding layer
        with tf.name_scope('embedding layer'):
            self.W = tf.Variable(tf.random_uniform([num_vocab, embedding_size],-1.0,1.0))
            self.embed_char = tf.nn.embedding_lookup(self.W,self.input)
            self.embed_char_expanded = tf.expand_dims(self.embed_char, -1) #在指定维度上增加一个维度

            pooled_outputs = []
            for filter_size in enumerate(filter_size_list):
                with tf.name_scope('convolution-maxpooling-%s'%filter_size):
                    filter_size = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_size,stddev=0.1))
                    b = tf.Variable(tf.truncated_normal(num_filters))
                    conv = tf.nn.conv2d(
                        self.embed_char_expanded,
                        filter=W,
                        strides=[1,1,1,1],
                        padding='VALID'
                        )
                    h = tf.nn.relu(tf.nn.bias_add(conv,b))
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_len-filter_size +1,1,1],
                        padding='VALID',
                        name='pool'
                        )
                    pooled_outputs.append(pooled)
                    num_filter_total = num_filters * len(filter_size)
                    self.h_pool = tf.concat(pooled_outputs,3)
                    self.h_pool_flatten = tf.reshape(self.h_pool,[-1,num_filter_total])
                with tf.name_scope('drop_out'):
                    self.drop = tf.nn.dropout(self.h_pool_flatten, self.hold_keep_hold)
                with tf.name_scope('output'):
                    W =  tf.get_variable('W',
                                         shape=[num_filter_total, num_class],
                                         initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1,shape=[num_class]), name='b')
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    self.scores = tf.nn.xw_plus_b(self.drop, W, b, name='scores')
                    self.predictions = tf.argmax(self.scores,1,name='predictions' )

                    with tf.name_scope('loss'):
                        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.output)
                        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
                    #accurracy
                    with tf.name_scope('accuracy'):
                        correct_predictions = tf.equal(self.predictions, tf.argmax(self.output,1))
                        self.accuray =  tf.reduce_mean(tf.cast(correct_predictions, 'float'),name='accuracy')











