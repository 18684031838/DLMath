import torch
import numpy
from torch.profiler import profile, record_function, ProfilerActivity

def load_data():
    t1 = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                           [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]])
    t2 = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]])
    t3 = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                           [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]])
    t4 = torch.tensor([[1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]])

    t5 = torch.tensor([[1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                           [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]])
    t6 = torch.tensor([[1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                           [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1]])
    t7 = torch.tensor([[0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
                           [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1]])
    t8 = torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
                           [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]])

    t9 = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]])

    t10 = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
                            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1]])

    t11 = torch.tensor([[0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
                            [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]])

    t12 = torch.tensor([[0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1], [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]])

    t13 = torch.tensor([[1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                            [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0]])

    t14 = torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]])

    t15 = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]])

    t16 = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]])

    result = torch.concat((t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16),dim=0)
    return result


def load_label():
    t01 = torch.tensor([1])
    t02 = torch.tensor([0])

    tx01 = t01.expand(32, -1)
    tx02 = t02.expand(32, -1)

    t11 = torch.tensor([0])
    t12 = torch.tensor([1])

    tx11 = t11.expand(32, -1)
    tx12 = t12.expand(32, -1)

    t1 = torch.cat((tx01,tx11),dim=0)
    t2 = torch.cat((tx02,tx12),dim=0)

    return t1,t2


def init_hidden_layer1_weight():
    w1 = torch.tensor([[0.49038557898997, 0.348475676796501, 0.0725879008695083,
                                    0.837472826850604, -0.0706798311519743, -3.6169369170322,
                                    -0.53557819719488, -0.0228584789393108, -1.71745249082217,
                                    -1.45563751579807, -0.555799932254451, 0.852476539980059]])
    b1 = -0.185002356132065

    return w1, b1


def init_hidden_layer2_weight():
    w2 = torch.tensor([[0.442372911956926, -0.536877487857221, 1.00782536916829,
                                    1.07196001297575, -0.732814485632708, 0.822959617857012,
                                    -0.453282364154155, -0.0138979392949318, -0.0274233258563056,
                                    -0.426670298661898, 1.87560275441379, -2.30528048189891]])
    b2 = 0.525676844318642

    return w2, b2


def init_hidden_layer3_weight():
    w3 = torch.tensor([[0.654393041569443, -1.38856820257739, 1.24648311661583,
                                    0.0572877158406771, -0.183237472237546, -0.74305066513479,
                                    -0.460930664925325, 0.331118557255208, 0.449470835925128,
                                    -1.29645372413246, 1.56850561324256, -0.470667153317658]])
    b3 = -1.16862269778991

    return w3, b3


def init_output_layer1_weight():
    w1 = torch.tensor([[0.3880031194962, 0.803384989025837, 0.0292864334994403]])
    b1 = -1.43803971240614
    return w1, b1


def init_output_layer2_weight():
    w2 = torch.tensor([[0.0254467679708455, -0.790397993881956, 1.55313793058729]])
    b2 = -1.37933790823328
    return w2, b2


def linear(x, w, b):
    return torch.sum(x * w, dim=1) + b


def sigmoid(x):
    result = 1 / (1 + torch.exp(-x))
    return result


def sigmoid_derivative(a):
    return a * (1 - a)


def cost(right_answer,predicate_result):
    t1,t2 = right_answer
    p1,p2 = predicate_result
    p1_reshape = torch.unsqueeze(p1,1)
    p2_reshape = torch.unsqueeze(p2,1)

    return ((t1-p1_reshape)**2+(t2-p2_reshape)**2)/2


def training(epochs,learning_rate):
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], profile_memory=True,record_shapes=True) as prof:
        # 加载输入层数据，为64张图片，每张图片为12个神经元
        x = load_data()
        # 每个输入神经元分别与隐藏层的3个神经元以不同的权重的偏置产生连接,这里初始化固定值是为了便于与书中excel对比分析，实际使用时，通常是随机使用正态分布的参数进行初始化
        w1, b1 = init_hidden_layer1_weight()
        w2, b2 = init_hidden_layer2_weight()
        w3, b3 = init_hidden_layer3_weight()

        for i in range(epochs):
            h1 = linear(x, w1, b1)
            a1 = sigmoid(h1)

            h2 = linear(x, w2, b2)
            a2 = sigmoid(h2)

            h3 = linear(x, w3, b3)
            a3 = sigmoid(h3)

            # 隐藏层神经元的误差
            d1 = sigmoid_derivative(a1)
            d2 = sigmoid_derivative(a2)
            d3 = sigmoid_derivative(a3)

            # 输出层结果
            ow1, ob1 = init_output_layer1_weight()
            ow2, ob2 = init_output_layer2_weight()

            o = torch.cat((a1.view(64, 1), a2.view(64, 1), a3.view(64, 1)), dim=1)
            z1 = linear(o, ow1, ob1)
            z2 = linear(o, ow2, ob2)

            # 预测的正解变量
            o1 = sigmoid(z1)
            o2 = sigmoid(z2)

            # 预测结果的激活函数求导
            ao1 = sigmoid_derivative(o1)
            ao2 = sigmoid_derivative(o2)

            # 代价函数计算结果
            right_answer = load_label()
            t1, t2 = right_answer
            c = cost(right_answer, (o1, o2))

            # 代价函数对求输出神经元的偏导数
            do1 = o1.view(64, 1) - t1
            do2 = o2.view(64, 1) - t2

            # 第3层的神经单元误差
            delta31 = ao1.view(64, 1) * do1.view(64, 1)
            delta32 = ao2.view(64, 1) * do2.view(64, 1)

            output_weight = torch.cat((ow1, ow2), dim=0).t()
            delta3 = torch.cat((delta31, delta32), dim=1)

            sum_weight_delta = (output_weight @ delta3.t()).t()

            hidden_layer_derivative = torch.cat((d1.view(64, 1), d2.view(64, 1), d3.view(64, 1)), dim=1)

            # 第二层神经元误差
            delta2 = sum_weight_delta * hidden_layer_derivative

            # 平方误差的偏导数
            # 隐藏层的偏导数
            delta21 = delta2[:, 0].view(64, 1)
            delta22 = delta2[:, 1].view(64, 1)
            delta23 = delta2[:, 2].view(64, 1)
            dc_hidden_layer1 = x * delta21.expand(-1, 12)
            dc_hidden_layer2 = x * delta22.expand(-1, 12)
            dc_hidden_layer3 = x * delta23.expand(-1, 12)

            # 总的损失函数对于第一个隐藏层的权重和偏置的梯度
            total_cost_derivative_hidden_layer1_weight = torch.sum(dc_hidden_layer1, dim=0)
            total_cost_derivative_hidden_layer2_weight = torch.sum(dc_hidden_layer2, dim=0)
            total_cost_derivative_hidden_layer3_weight = torch.sum(dc_hidden_layer3, dim=0)

            total_cost_derivative_hidden_layer1_bias = torch.sum(delta21, dim=0)
            total_cost_derivative_hidden_layer2_bias = torch.sum(delta22, dim=0)
            total_cost_derivative_hidden_layer3_bias = torch.sum(delta23, dim=0)

            # 损失函数对于输入层的权重和偏置的梯度
            # 1.构造隐藏层输出结果矩阵
            hidden_layer_output = torch.concat((a1.view(64, 1), a2.view(64, 1), a3.view(64, 1)), dim=1)

            # 2.通过第3层神经元误差矩阵的转置与隐藏层输出结果矩阵相乘求出损失函数对输出层的权重和偏置的导数
            # 将 delta3 转换为与 hidden_layer_output 进行矩阵相乘

            delta3_transformed = delta3.view(64, 2, 1)

            # 转换 hidden_layer_output 以进行矩阵相乘,得到代价函数对于输出层权重的偏置
            hidden_layer_output_transformed = hidden_layer_output.view(64, 1, 3)
            cost_derivative_output_layer = delta3_transformed @ hidden_layer_output_transformed
            total_cost = torch.sum(c, dim=0)
            print(f"第{i+1}轮训练损失：{total_cost}")

            # 上一轮的权重减去隐藏层的偏导数乘以学习率
            w1 = w1 - torch.sum(dc_hidden_layer1,dim=0).view(1,12) * learning_rate
            b1 = b1 - torch.sum(delta21,dim=0)

            w2 = w2 - torch.sum(dc_hidden_layer2,dim=0).view(1,12) * learning_rate
            b2 = b2 - torch.sum(delta22, dim=0)

            w3 = w3 - torch.sum(dc_hidden_layer3,dim=0).view(1,12) * learning_rate
            b3 = b3 - torch.sum(delta23, dim=0)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


