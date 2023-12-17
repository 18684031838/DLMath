# 将深度学习中的数学书中的关于误差反向传播的实例，改写为pytorch框架的实现并输出中间变量和梯度计算的数据，
# 与Excel计算过程比较，了解pytorch实现的内部原理和内部计算过程
import torch

import DLMathBP as bp


if __name__ == '__main__':
    bp.training(50,0.2)
    print("ok")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
