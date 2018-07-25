# -*- coding: utf-8 -*-
import tensorflow as tf
# 导入图片数据集
from tensorflow.examples.tutorials.mnist import input_data

# 输入层节点数784，一张照片时[1行,784列]数组，数值是长宽28*28像素的乘积，把照片上的每个像素点输入到一个节点
INPUT_NODE = 784
# 输出层节点数10，一张照片时[1行,10列]，10个节点中某个节点有输出，代表预测图片数据集中要区分的0~9共10个数字
OUTPUT_NODE = 10
# 隐藏层节点数
LAYER1_NODE = 500
# 梯度下降算法中使用的抽取样本数量。不能计算所有数据，耗时太长。
BATCH_SIZE = 100
# 基础学习率
LEARNING_RATE_BASE = 0.8
# 学习率的衰减率
LEARNING_RATE_DECAY = 0.99
# 正则化率，描述模型复杂程度的正则化项在损失函数中的系数
REGULARIZATION_RATE = 0.0001
# 训练轮数
TRAINING_STEPS = 30000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

# 给定输入和所有参数，计算前向传播结果。
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值。
    if avg_class == None:
        # 使用relu激活函数去线性化，f(x)=max(x, 0)，x>=0时,f(x)=x, x<0时,f(x)=0，将每个节点的输出通过非线性函数，
        # 整个模型就不是线性的了。
        # matmul函数 矩阵乘法，隐藏层1 = [输入层] * [权重1] + 偏置项1
        # 两个矩阵相乘，矩阵1的列数要等于矩阵2的行数，[1, 784]*[784, 500]=[1, 500] 
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 矩阵乘法，return前向传播结果 = [隐藏层1] * [权重2] + 偏置项2
        # [1, 500]*[500, 10]=[1 ,10] 返回输出层 1行10列矩阵，每一列代表预测0~9其中的一个数字
        return tf.matmul(layer1, weights2) + biases2
    # 首先使用avg_class.average函数计算得出变量的滑动平均值，然后用矩阵乘法，最后return前向传播结果。
    else:
        layer1 = tf.nn.relu(
                tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
    
def train(mnist):
    # tf模块中的变量类型叫张量，必须定义张量的三个属性：类型(实数/整数/布尔型/复数)，纬度(shape=[2, 3]二维数组，2行3列矩阵)，
    # 名字(标识符，代表如何计算出来)
    # 三维坐标(1,2,3)表示在坐标轴x,y,z方向上的位置。类似的，张量的维度表示不同维度方向上的分布。列表只有长短所以是一维数组，
    # 矩阵有行、列的方向上的分布所以是二维数组，三维数组可以理解成x,y,z坐标轴上的数据分布。
    # placeholder函数相当于定义了一个位置，这个位置中的数据在程序运行时再指定
    # x输入层，float32浮点数，0行784列矩阵，名字"x-input"
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    # y_输出层，float32浮点数，0行10列矩阵，名字"y-input"
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")
    # 生成隐藏层的参数，权重1。Variable函数保存和更新参数，truncated_normal生成正态分布的随机数，784行500列矩阵，标准差0.1
    weights1 = tf.Variable(
            tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    # 偏置项1。constant生成给定值的常量。值为0.1，纬度500。
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数，权重2
    weights2 = tf.Variable(
            tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    # 偏置项2
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    # 调用自定义inference函数。滑动平均的类为None，不使用参数的滑动平均值。
    y = inference(x, None, weights1, biases1, weights2, biases2)
    # 定义存储训练轮数的变量，初始值为0，不可训练。
    global_step = tf.Variable(0, trainable=False)
    # 滑动平均衰减率MOVING_AVERAGE_DECAY=0.99，训练轮数global_step，初始化滑动平均类。
    # 给定训练轮数可以加快训练早期变量的更新速度。
    # ExponentialMovingAverage函数，对每个变量会维护一个影子变量，影子变量=衰减率*影子变量+(1-衰减率)*待更新的变量，
    # 减率越大，第二项值越趋近于0，模型越稳定。
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    # 在这些参数的变量上使用滑动平均。
    variable_averages_op = variable_averages.apply(
            tf.trainable_variables())
     # 调用自定义inference函数，计算使用了滑动平均之后的前向传播结果。
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    # softmax函数把输出变成0~1(0不发生，1一定发生)之间的概率分布，公式是softmax(y)i=e^yi/(e^y1+e^y2+...+e^yi)，
    # cross_entropy交叉熵函数−∑x(p(x)ln(q(x))，刻画两个概率分布y输出结果, y_标准答案之间的距离，
    # argmax(y_, 1)函数表示在第1个维度选取最大值。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前训练轮数中所有样例的交叉熵平均值，评价正确答案和预测值之间的差距。
    # reduce_mean均方误差损失函数，(∑(正确答案−预测值)^2)/n
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 返回L2正则化函数，防止过度拟合，一个模型过为复杂后，它会很好的“记忆”每一个
    # 训练数据中随机噪音的部分而忘记了要去“学习”训练数据中通用的趋势。公式∑(每个权重^2)*复杂损失在总损失中的比例即正则化率
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 利用上面的函数计算正则化损失，权重作为参数传入
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失+正则化损失
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率，控制了参数每次更新的幅度。如果幅度过大，参数会在极优值的两侧来回移动。
    # 公式 每一轮使用的学习率=初始学习率*衰减系数^(训练轮数/衰减速度)
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    # 最重要的一步，GradientDescentOptimizer函数使用梯度下降算法来优化损失函数的参数loss。返回一个[梯度，变量]对应的列表。
    # 梯度下降的作用寻找参数β对应损失函数f(β)的最小值，此时β处的f(β)导数为0。公式 新参数β'=原参数β-学习率*(f(β)偏导/β偏导)
    # 偏导数 二元二次x,y方程f(x,y)=x^2+3xy+y^2对x求偏导，把y当成常数然后对x求导数，结果f'(x)=2x+3y
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                .minimize(loss, global_step=global_step)
    # 每次完成一轮训练时需要通过反向传播更新所有的参数和它们的滑动平均值，返回train_op。传入的参数是上面的变量。
    # control_dependencies函数用来控制计算流图的，给图中的这些计算指定顺序。
    # no_op函数不做任何事情，只占位控制执行流程
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")
    # 检验使用了滑动平均模型的前向传播结果是否正确，arg_max(average_y, 1)函数计算每一个样例的预测答案。
    # 使用样本BATCH_SIZE = 100后就是导入100张照片，average_y是一个[100行，10列]的矩阵，每一行表示一张
    # 照片的前向传播结果，第二个参数"1"表示在"行"的维度中选取最大值(舍弃无关的输出节点，得到预测值），
    # 得到一个长度为[100行,1列]矩阵，表示每张照片对应的数字识别结果。
    # equal函数判断上面两个张量的每个维度是否相等，返回True/False。就是判断图片是否预测成功
    corrcet_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
    # 得到正确率，cast函数把上面的布尔值转换成浮点数，reduce_mean函数计算平均值
    accuracy = tf.reduce_mean(tf.cast(corrcet_prediction, tf.float32))
    # session会话的方式开始训练过程，结束后自动回收资源
    with tf.Session() as sess:
        # global_variables_initializer函数，获取所有变量然后初始化，自动处理变量之间的依赖关系
        # 初始化和会话运行都要用run函数开始执行
        tf.global_variables_initializer().run()
        # 准备验证数据，导入验证照片集中的照片和答案
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        # 准备测试数据，导入测试照片集中的照片和答案
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        # 循环训练网络模型
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据集上的对比结果
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                # 打印每轮训练后验证的正确率
                print("After %d training steps, validation accuracy using average model is %g" % (i, validate_acc))
            # 生成这一轮使用的一个batch的训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 运行训练过程，更新传入的参数和它们的滑动平均值，train_op
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练结束后，在测试数据集上检测照片和答案的对比结果
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        # 打印预测的最终正确率
        print("After %d training steps, test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))
        
def main(argv=None):
    # 自动加载mnist数据集，并划分成train, validation, test数据集
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    # 开始训练函数train()
    train(mnist)

if __name__ == "__main__":
    # 调用上面的main()函数
    tf.app.run()
            
        
      
        
        