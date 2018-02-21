import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
from PIL import Image, ImageFilter
# 下载训练和测试数据
mnist = input_data.read_data_sets('C:/da', one_hot = True)

# 创建session
sess = tf.Session()

# 占位符
x = tf.placeholder(tf.float32, shape=[None, 784]) # 每张图片28*28，共784个像素
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # 输出为0-9共10个数字，其实就是把图片分为10类

# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 使用截尾正态分布的随机数初始化权重，标准偏差是0.1（噪音）
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape) # 使用一个小正数初始化偏置，避免出现偏置总为0的情况
    return tf.Variable(initial)

# 卷积和集合
def conv2d(x, W): # 计算2d卷积
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x): # 计算最大集合
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32]) # 为每个5*5小块计算32个特征
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1]) # 将图片像素转换为4维tensor，其中二三维是宽高，第四维是像素
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集层
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 创建1024个神经元对整个图片进行处理
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 退出（为了减少过度拟合，在读取层前面加退出层，仅训练时有效）
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 读取层（最后我们加一个像softmax表达式那样的层）
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 预测类和损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) # 计算偏差平均值
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 每一步训练

# 评估
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#init_op = tf.initialize_all_variables()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0}, session = sess) # 每10次训练计算一次精度
        print("步数 %d, 精度 %g"%(i, train_accuracy))
#    if train_accuracy >= 1:
#        break
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, session = sess)


saver = tf.train.Saver()

save_path = saver.save(sess, "./f/aki.ckpt")
print ("Model saved in file: ", save_path)
