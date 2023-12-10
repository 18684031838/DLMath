# 将深度学习中的数学书中的关于误差反向传播的实例，改写为pytorch框架的实现并输出中间变量和梯度计算的数据，
# 与Excel计算过程比较，了解pytorch实现的内部原理和内部计算过程
import torch

import DLMathBP as bp



if __name__ == '__main__':
    # 加载输入层数据，为64张图片，每张图片为12个神经元
    x = bp.load_data()

    # 每个输入神经元分别与隐藏层的3个神经元以不同的权重的偏置产生连接,这里初始化固定值是为了便于与书中excel对比分析，实际使用时，通常是随机使用正态分布的参数进行初始化
    w1,b1 = bp.init_hidden_layer1_weight()
    h1 = bp.linear(x,w1,b1)
    a1 = bp.sigmoid(h1)

    w2, b2 = bp.init_hidden_layer2_weight()
    h2 = bp.linear(x, w2, b2)
    a2 = bp.sigmoid(h2)

    w3, b3 = bp.init_hidden_layer3_weight()
    h3 = bp.linear(x, w3, b3)
    a3 = bp.sigmoid(h3)

    # 隐藏层神经元的误差
    d1 = bp.sigmoid_derivative(a1)
    d2 = bp.sigmoid_derivative(a2)
    d3 = bp.sigmoid_derivative(a3)

    # 输出层结果
    ow1,ob1 = bp.init_output_layer1_weight()
    ow2,ob2 = bp.init_output_layer2_weight()

    o = torch.cat((a1.view(64,1), a2.view(64,1), a3.view(64,1)), dim=1)
    z1 = bp.linear(o, ow1, ob1)
    z2 = bp.linear(o, ow2, ob2)

    # 预测的正解变量
    o1 = bp.sigmoid(z1)
    o2 = bp.sigmoid(z2)

    # 预测结果的激活函数求导
    ao1 = bp.sigmoid_derivative(o1)
    ao2 = bp.sigmoid_derivative(o2)

    # 代价函数计算结果
    right_answer = bp.load_label()
    t1,t2 =right_answer
    c = bp.cost(right_answer,(o1,o2))

    # 代价函数对求输出神经元的偏导数
    do1 = o1.view(64,1) - t1
    do2 = o2.view(64,1) - t2

    # 第3层的神经单元误差
    delta31 = ao1.view(64, 1) * do1.view(64, 1)
    delta32 = ao2.view(64,1) * do2.view(64,1)

    output_weight = torch.cat((ow1, ow2), dim=0).t()
    delta3 = torch.cat((delta31,delta32),dim=1)

    sum_weight_delta = (output_weight @ delta3.t()).t()

    hidden_layer_derivative = torch.cat((d1.view(64, 1), d2.view(64, 1), d3.view(64, 1)),dim=1)

    # 第二层神经元误差
    delta2 = sum_weight_delta * hidden_layer_derivative

    # 平方误差的偏导数
    # 隐藏层1的偏导数
    delta21 = delta2[:, 0].view(64, 1)
    delta22 = delta2[:, 1].view(64, 1)
    delta23 = delta2[:, 2].view(64, 1)
    dc_hidden_layer1 = x * delta21.expand(-1, 12)
    dc_hidden_layer2 = x * delta22.expand(-1, 12)
    dc_hidden_layer3 = x * delta23.expand(-1, 12)

   # 总的损失函数对于第一个隐藏层的权重和偏置的梯度
    total_cost_derivative_hidden_layer1_weight = torch.sum(dc_hidden_layer1,dim=0)
    total_cost_derivative_hidden_layer2_weight = torch.sum(dc_hidden_layer2, dim=0)
    total_cost_derivative_hidden_layer3_weight = torch.sum(dc_hidden_layer3, dim=0)

    total_cost_derivative_hidden_layer1_bias = torch.sum(delta21, dim=0)
    total_cost_derivative_hidden_layer2_bias = torch.sum(delta22, dim=0)
    total_cost_derivative_hidden_layer3_bias = torch.sum(delta23, dim=0)

    # 损失函数对于输入层的权重和偏置的梯度
    # 1.构造隐藏层输出结果矩阵
    hidden_layer_output = torch.concat((a1.view(64, 1), a2.view(64, 1), a3.view(64, 1)),dim=1)

    # 2.通过第3层神经元误差矩阵的转置与隐藏层输出结果矩阵相乘求出损失函数对输出层的权重和偏置的导数
    # 将 delta3 转换为与 hidden_layer_output 进行矩阵相乘

    delta3_transformed = delta3.view(64, 2, 1)

    # 转换 hidden_layer_output 以进行矩阵相乘
    hidden_layer_output_transformed = hidden_layer_output.view(64, 1, 3)
    result =  delta3_transformed @ hidden_layer_output_transformed



    print("ok")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
