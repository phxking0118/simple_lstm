import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import os
# factor_list = set(range(1,192))
# unrealized_fac = {19,30,75,143,149,165,181,182,183}
# factor_list = list(factor_list - unrealized_fac )
def path_setting(root_path,path_name_list):
    for name in path_name_list:
        if os.path.exists(root_path+name):
            print(name+'exists')
        else:
            os.mkdir(root_path+name)
            print(name+'made')



##定义激活函数,可根据需要选取不同的损失函数
##阶跃函数
def leap(x):
    if x < 0 or x == 0:
        return 0.0
    elif np.isnan(x):
        return np.nan
    else:
        return 1.0


leap_vec = np.vectorize(leap)


###阶跃函数导数
def dleap(x):
    if x < 0 or x > 0:
        return 0.0
    else:
        return np.nan


dleap_vec = np.vectorize(dleap)


###sigmoid函数
def sigmoid(x):
    if x > (-700.0):
        y = 1 / (1 + np.exp(-x))
        return y
    elif np.isnan(x):
        return np.nan
    else:
        # 在自变量超出范围(-700,inf)时取值为-700对应的值
        return 1 / (1 + np.exp(-700.0))


sigmoid_vec = np.vectorize(sigmoid)


##sigmoid函数的导数
def dsigmoid(x):
    if x > (-700.0):
        y = 1 / (1 + np.exp(-x))
        y = y * (1 - y)
        return y
    elif x == (-700.0) or np.isnan(x):
        return np.nan
    else:
        return 0.0


dsigmoid_vec = np.vectorize(dsigmoid)


##
###relu函数
def relu(x):
    if x < 0:
        return 0.0
    elif np.isnan(x):
        return np.nan
    else:
        return x
    # 下述函数也可以作为relu函数的向量化函数


# relu_vec=np.vectorize(relu)
# 定义关于矩阵的relu函数
def relu_vec(X):
    X[np.isnan(X)] = np.nan
    X[X < 0] = 0
    return X


##relu函数的导数
def drelu(x) -> float:
    if x < 0:
        return 0.0
    elif x == 0 or np.isnan(x):
        return np.nan
    else:
        return 1.0
    # 下述函数也可以作为relu导数的向量化函数


# drelu_vec=np.vectorize(drelu)
def drelu_vec(X):
    X[np.isnan(X)] = np.nan
    X[X > 0] = 1
    X[X <= 0] = 0
    return X


##tanh激活函数
# 自变量范围取为-700或者700
# -700对应的tanh
tanh_700_ = (np.exp(-700.0) - np.exp(700.0)) / ((np.exp(-700.0) + np.exp(700.0)))
# 700对应的tanh函数值
tanh_700 = -tanh_700_


def tanh(x):
    if np.isnan(x):
        return np.nan
    elif x <= (-700.0):
        return (tanh_700_)
    elif x >= (700.0):
        return (tanh_700)
    else:
        return (np.exp(x) - np.exp(-x)) / ((np.exp(x) + np.exp(-x)))


tanh_vec = np.vectorize(tanh)


##tanh的导数
def dtanh(x):
    if np.isnan(x):
        return np.nan
    elif (x <= (-700.0) or x >= (700.0)):
        return 0
    else:
        return 4 / ((np.exp(x) + np.exp(-x)) * (np.exp(x) + np.exp(-x)))


dtanh_vec = np.vectorize(dtanh)


# 定义一种新的矩阵乘法
# 对于含有大量nan的矩阵，先将nan变为0，再做乘法作为输出值
# 将nan转化为0
def dot_np(X, Y):
    X_1 = np.nan_to_num(X)
    Y_1 = np.nan_to_num(Y)
    return np.dot(X_1, Y_1)


# 定义二维权值矩阵与三维输入矩阵的乘法
# 例如一个矩阵为(85,2700,1588),另一矩阵为(20,85)，利用该乘积可以得到(20,2700,1588)的矩阵
def mul(x, y):
    X = x.T
    Y = y.T
    result = (np.dot(Y, X)).T
    return result

#时间维度上选取8个时刻
def factor_made(rootpath,lens,days_span):
    for i in range(0,lens,5):
        #选取85个因子
        for k in range(1,9):
            #导入因子文件
            factor_matrix=np.load(rootpath+'factors_value\\%d.npy'%(k))
            #对每个导入的矩阵提取从第i天开始1500天的数据
            add_list=factor_matrix[i:days_span+i,:]
            #将1500天，1588只股票的数据展成一个向量，便于处理
            add_list=add_list.flatten().reshape(1,add_list.size)
            #将85个因子展开得到的向量按顺序拼接起来，最终会得到(85,1500*1588)的二维矩阵
            if k==1:
                time_list=add_list.copy()
            else:
                time_list=np.vstack((time_list,add_list.copy()))
        time_list=np.array(time_list)
        #对得到的矩阵进行处理
        i=i/5
        np.save(rootpath+'start_dict\\0_%d.npy'%i,time_list)
    return 'process finished'

# 注：下述函数中的参数大小均与pdf中的内容匹配，请参照pdf
# 举例对象的矩阵维度，与pdf中的例子对应
# 该代码对应的神经网络结构，与pdf中相比，pdf中使用sigmoid函数的地方，该代码中均变成了relu函数

# 定义lstm中每两层之间的正向运算函数
# 处理对象包括非输出层的其余所有层，及pdf中的m个隐藏层

# 输入参数的含义：
# 1. sub_c_0为每个lstm层的初始细胞状态矩阵，一共含有m个矩阵.以Pdf中的例子来看，sub_c_0[0],sub_c_0[1]分别
# 表示第1、2个隐藏层的初始化细胞状态矩阵，维数分别为(20,2382000),(5,2382000)
# 2. sub_h_0为每个lstm层的初始隐藏层的输出矩阵，一共含有m个矩阵.以Pdf中的例子来看，sub_h_0[0],sub_h_0[1]分别
# 表示第1、2个隐藏层的初始化输出矩阵，维数分别为(20,2382000),(5,2382000)
# 3. sub_W是一个字典。长度对应隐藏层的个数,即m，其中的键k对应第(k+1)个隐藏层,其中每个键对应的值也为一个字典，该字典中共有8个元素，每个元素
# 为一个矩阵，对应lstm层的8个权值矩阵，顺序与pdf中的顺序相对应
# 4. bias是一个字典，长度对应隐藏层的个数,即m，其中的键k对应第(k+1)个隐藏层，其中每个键对应的值也为一个字典，该字典中共有4个元素，每个元素
# 为一个向量，(维数为该隐藏层神经元个数，1），对应lstm层的4个偏差项矩阵，顺序与pdf中的顺序相对应
# 5. time_len为该神经网络中用的时间序列长度，例子中即为8
# 6. layer_len为该神经网络中隐藏层的个数
def forward(rootpath, sub_c_0, sub_h_0, sub_W, bias, time_len, layer_len):
    # 对隐藏层进行遍历
    for k in range(layer_len):
        # 先对第一个时刻进行计算
        # 对遗忘门进行操作
        # 导入第一个时刻输入层的矩阵，作为第一个隐藏层的输入阵
        sub_start = np.load(rootpath + 'start_dict\\%d_0.npy' % k)
        # 按pdf中公式进行计算，此处未计算偏差项
        net_forget = np.dot(sub_W[k][0], sub_h_0[k]) + np.dot(sub_W[k][1], sub_start)
        # 生成偏差矩阵，对于每个样本，在每一层中使用的偏差矩阵相同，下式中net_forget.shape[1]为该层网络中的样本总数
        bias_one = np.ones((bias[k][0].shape[0], net_forget.shape[1]))
        # 加入偏差项，并用激活函数进行作用
        sub_forget = relu_vec(net_forget + bias[k][0] * bias_one)
        # 保存结果，两个%d,前面的表示层数，后面的表示时间，第一个时刻记为0
        np.save(rootpath + 'forget_dict\\%d_%d.npy' % (k, 0), sub_forget)
        # 对输入门进行类似的操作
        net_input = np.dot(sub_W[k][2], sub_h_0[k]) + np.dot(sub_W[k][3], sub_start)
        sub_input = relu_vec(net_input + bias[k][1] * bias_one)
        np.save(rootpath + 'input_dict\\%d_%d.npy' % (k, 0), sub_input)
        # 对候选门进行类似操作，但是候选门激活函数使用tanh
        net_candidate = np.dot(sub_W[k][4], sub_h_0[k]) + np.dot(sub_W[k][5], sub_start)
        sub_candidate = tanh_vec(net_candidate + bias[k][2] * bias_one)
        np.save(rootpath + 'candidate_dict\\%d_%d.npy' % (k, 0), sub_candidate)
        # 对输出门进行类似操作
        net_output = np.dot(sub_W[k][6], sub_h_0[k]) + np.dot(sub_W[k][7], sub_start)
        sub_output = relu_vec(net_output + bias[k][3] * bias_one)
        np.save(rootpath + 'output_dict\\%d_%d.npy' % (k, 0), sub_output)
        # 利用已有结果计算细胞状态矩阵
        sub_state = sub_forget * sub_c_0[k] + sub_input * sub_candidate
        np.save(rootpath + 'state_dict\\%d_%d.npy' % (k, 0), sub_state)
        # 计算本单元最终输出
        sub_last_output = tanh_vec(sub_state) * sub_output
        # 因为本单元的输出作为下一个隐藏层同一时刻的输入，所以也存储在start_dict中
        # start_dict中的(k+1,0)即表示第一个时刻第k+1隐藏层的输入，又表示第k个隐藏层的输出
        np.save(rootpath + 'start_dict\\%d_%d.npy' % (k + 1, 0), sub_last_output)

        # 对之后的时刻进行循环运算，原理同第一个时刻
        # 改变之处在于第一个时刻对应的c_0,h_0,在后续步走中为上一个时刻对应的c和h矩阵
        for i in range(1, time_len):
            # 导入计算需要的矩阵
            sub_start = np.load(rootpath + 'start_dict\\%d_%d.npy' % (k, i))
            sub_h = np.load(rootpath + 'start_dict\\%d_%d.npy' % (k + 1, i - 1))
            # 对遗忘门进行计算
            net_forget = np.dot(sub_W[k][0], sub_h) + np.dot(sub_W[k][1], sub_start)
            sub_forget = relu_vec(net_forget + bias[k][0] * bias_one)
            np.save(rootpath + 'forget_dict\\%d_%d.npy' % (k, i), sub_forget)
            # 对输入门进行计算
            net_input = np.dot(sub_W[k][2], sub_h) + np.dot(sub_W[k][3], sub_start)
            sub_input = relu_vec(net_input + bias[k][1] * bias_one)
            np.save(rootpath + 'input_dict\\%d_%d.npy' % (k, i), sub_input)
            # 对候选门进行计算
            net_candidate = np.dot(sub_W[k][4], sub_h) + np.dot(sub_W[k][5], sub_start)
            sub_candidate = tanh_vec(net_candidate + bias[k][2] * bias_one)
            np.save(rootpath + 'candidate_dict\\%d_%d.npy' % (k, i), sub_candidate)
            # 对输出门进行计算
            net_output = np.dot(sub_W[k][6], sub_h) + np.dot(sub_W[k][7], sub_start)
            sub_output = relu_vec(net_output + bias[k][3] * bias_one)
            np.save(rootpath + 'output_dict\\%d_%d.npy' % (k, i), sub_output)
            # 利用已有结果计算细胞状态矩阵
            sub_c = np.load(rootpath + 'state_dict\\%d_%d.npy' % (k, i - 1))
            sub_state = sub_forget * sub_c + sub_input * sub_candidate
            np.save(rootpath + 'state_dict\\%d_%d.npy' % (k, i), sub_state)
            # 计算本单元最终输出
            sub_last_output = tanh_vec(sub_state) * sub_output
            np.save(rootpath + 'start_dict\\%d_%d.npy' % (k + 1, i), sub_last_output)


# 网络的最后一个隐藏层到输出层计算函数
# 因为输出层只是CNN，不是循环网络，所以与上面计算方法不同
# 该函数的输入参数：
# 1. sub_W_y 表示最后一个隐藏层到输出层的权值矩阵，对应pdf中的例子，维数为(1,5)
# 2. sub_bias_y 表示一个偏苯的偏差项，为一个列向量,对应pdf中的例子,维数为(1,1)
# 3. time_len 为时间长度
# 4. layer_len  为隐藏层的个数
def forward_last_layer(rootpath, sub_W_y, sub_bias_y, time_len, layer_len):
    # 导入每个时刻的输入矩阵
    sub_start = np.load(rootpath + 'start_dict\\%d_%d.npy' % (layer_len, 0))
    # 每个样本对应的偏差项相同，所以后续利用一个样本的偏差项生成全部样本的偏差项矩阵
    bias_one = np.ones((sub_bias_y.shape[0], sub_start.shape[1]))
    # 对时刻进行遍历
    for i in range(time_len):
        # 导入输出层的输入矩阵
        sub_start = np.load(rootpath + 'start_dict\\%d_%d.npy' % (layer_len, i))
        net_end = np.dot(sub_W_y, sub_start)
        # 加入偏差项，并作用激活函数
        sub_end = relu_vec(net_end + sub_bias_y * bias_one)
        # 为了统一计算，将输出层的输出记为第（m+1)个隐藏层的输入（假设共有m个隐藏层）
        np.save(rootpath + 'start_dict\\%d_%d.npy' % (layer_len + 1, i), sub_end)


# 反向计算梯度并更新权值矩阵和偏差项
# 对于最优一个隐藏层与输出层之间的权值矩阵和偏差项，在主循环中单独计算
# 该函数用于计算倒数第二个隐藏层与最后一个隐藏层之间的权值矩阵与偏差项更新
# 该函数输入的参数为：
# 1. sub_c_0为最后一个lstm层的初始细胞状态矩阵，以Pdf中的例子来看，sub_c_0
# 表示第2个隐藏层的初始化细胞状态矩阵，维数为(5,2382000)
# 2. sub_h_0为最后一个lstm层的初始隐藏层的输出矩阵，以Pdf中的例子来看，sub_h_0
# 表示第2个隐藏层的初始化输出矩阵，维数分别为(5,2382000)
# 3. sub_W是倒数第二个隐藏层向最后一个隐藏层传递时，对应lstm层的8个权值矩阵组成的字典，顺序与pdf中的顺序相对应
# 4. sub_W_y为最后一个隐藏层向输出层传递时的权值矩阵
# 5. sub_bias是一个字典，该字典中共有4个元素，每个元素为一个向量，(维数为该隐藏层神经元个数，1），对应lstm层的4个偏差项矩阵，顺序与pdf中的顺序相对应
# 6. time_len为该神经网络中用的时间序列长度，例子中即为8
# 7. layer_len为该神经网络中隐藏层的个数
def backward_delta_last(rootpath, sub_c_0, sub_h_0, sub_W, sub_W_y, sub_bias, time_len, layer_len,learning_rate):
    # 权值矩阵sub_W的更新字典，顺序与sub_W中的顺序一一对应,该字典含8个元素
    sub_update_W = {}
    # sub_W的梯度字典，顺序与sub_W中的一一对应，该字典含8个元素
    sub_delta_W = {}
    # 偏差项sub_bias的更新字典，顺序与sub_bias中的一一对应，该字典含8个元素
    sub_update_bias = {}
    # 偏差项sub_bias的梯度字典，顺序与sub_bias中的一一对应，该字典含8个元素
    sub_delta_bias = {}
    # 对于sub_delta_W,sub_delta_bias赋初值0
    for i in range(8):
        sub_delta_W[i] = 0
    for i in range(4):
        sub_delta_bias[i] = 0
    # 对应pdf中的计算公式，在反向更新时，涉及到时刻(time_len的数据，对此数据，统一按0处理
    # 所以在此处补充0矩阵，0矩阵大小与这一隐藏层的sub_forget大小相同，所以导入以创建权值矩阵
    sub_forget = np.load(rootpath + 'forget_dict\\%d_%d.npy' % (layer_len - 1, 0))
    sub_add_zeros = np.zeros((sub_forget.shape[0], sub_forget.shape[1]))
    # 补充0矩阵，并保存
    np.save(rootpath + 'delta_forget_dict\\%d.npy' % (time_len), sub_add_zeros)
    np.save(rootpath + 'delta_input_dict\\%d.npy' % (time_len), sub_add_zeros)
    np.save(rootpath + 'delta_candidate_dict\\%d.npy' % (time_len), sub_add_zeros)
    np.save(rootpath + 'delta_output_dict\\%d.npy' % (time_len), sub_add_zeros)
    np.save(rootpath + 'forget_dict\\%d_%d.npy' % (layer_len - 1, time_len), sub_add_zeros)
    sub_epsilon_state = sub_add_zeros.copy()

    # 在时间维度上进行逆向遍历
    for i in range(time_len - 1, -1, -1):
        # 参考pdf中的计算公式,导入需要的矩阵
        # 下式为输出层的残差矩阵
        sub_delta_end = np.load(rootpath + 'delta_end_dict\\%d.npy' % i)
        # 下式为本隐藏层已经计算得到的下一时刻的残差矩阵
        sub_delta_forget = np.load(rootpath + 'delta_forget_dict\\%d.npy' % (i + 1))
        sub_delta_input = np.load(rootpath + 'delta_input_dict\\%d.npy' % (i + 1))
        sub_delta_candidate = np.load(rootpath + 'delta_candidate_dict\\%d.npy' % (i + 1))
        sub_delta_output = np.load(rootpath + 'delta_output_dict\\%d.npy' % (i + 1))

        # 根据pdf中的公式进行计算
        sub_epsilon_last_output = np.dot(sub_W_y.T, sub_delta_end) + np.dot(sub_W[0].T, sub_delta_forget) \
                                  + np.dot(sub_W[2].T, sub_delta_input) + np.dot(sub_W[4].T,
                                                                                 sub_delta_candidate) + np.dot(sub_W[6].T,
                                                                                                               sub_delta_output)

        sub_output = np.load(rootpath + 'output_dict\\%d_%d.npy' % (layer_len - 1, i))
        sub_state = np.load(rootpath + 'state_dict\\%d_%d.npy' % (layer_len - 1, i))
        sub_forget_plus1 = np.load(rootpath + 'forget_dict\\%d_%d.npy' % (layer_len - 1, i + 1))
        sub_candidate = np.load(rootpath + 'candidate_dict\\%d_%d.npy' % (layer_len - 1, i))
        sub_input = np.load(rootpath + 'input_dict\\%d_%d.npy' % (layer_len - 1, i))
        sub_forget = np.load(rootpath + 'forget_dict\\%d_%d.npy' % (layer_len - 1, i))
        # 对于本时刻前一时刻的细胞状态矩阵，若此时为第一个细胞状态，对应的矩阵则为初始化值sub_c_0
        if (i > 0):
            sub_state_minus1 = np.load(rootpath + 'state_dict\\%d_%d.npy' % (layer_len - 1, i - 1))
        else:
            sub_state_minus1 = sub_c_0

        # 参考具体公式进行计算，在计算的同时释放内存
        sub_epsilon_state = sub_epsilon_last_output * sub_output * dtanh_vec(sub_state) \
                            + sub_epsilon_state * sub_forget_plus1
        sub_delta_forget = sub_epsilon_state * sub_state_minus1 * drelu_vec(sub_forget)
        np.save(rootpath + 'delta_forget_dict\\%d.npy' % (i), sub_delta_forget)
        sub_delta_input = sub_epsilon_state * sub_candidate * drelu_vec(sub_input)
        np.save(rootpath + 'delta_input_dict\\%d.npy' % (i), sub_delta_input)
        sub_delta_candidate = sub_epsilon_state * sub_input * (1 - sub_candidate * sub_candidate)
        np.save(rootpath + 'delta_candidate_dict\\%d.npy' % (i), sub_delta_candidate)
        sub_delta_output = sub_epsilon_last_output * relu_vec(sub_state) * drelu_vec(sub_output)
        np.save(rootpath + 'delta_output_dict\\%d.npy' % (i), sub_delta_input)

        # 在得到残差之后，计算sub_W的梯度
        # 若为第一个时刻，要用初始化参数sub_h_0
        if i > 0:
            sub_start = np.load(rootpath + 'start_dict\\%d_%d.npy' % (layer_len, i - 1))
        else:
            sub_start = sub_h_0
        sub_start_minus = np.load(rootpath + 'start_dict\\%d_%d.npy' % (layer_len - 1, i))
        sub_delta_W[0] += dot_np(sub_delta_forget, sub_start.T) / sub_start.shape[1]
        sub_delta_W[1] += dot_np(sub_delta_forget, sub_start_minus.T) / sub_start_minus.shape[1]
        sub_delta_W[2] += dot_np(sub_delta_input, sub_start.T) / sub_start.shape[1]
        sub_delta_W[3] += dot_np(sub_delta_input, sub_start_minus.T) / sub_start_minus.shape[1]
        sub_delta_W[4] += dot_np(sub_delta_candidate, sub_start.T) / sub_start.shape[1]
        sub_delta_W[5] += dot_np(sub_delta_candidate, sub_start_minus.T) / sub_start_minus.shape[1]
        sub_delta_W[6] += dot_np(sub_delta_output, sub_start.T) / sub_start.shape[1]
        sub_delta_W[7] += dot_np(sub_delta_output, sub_start_minus.T) / sub_start_minus.shape[1]

        # 对于偏差项，由于不同样本对应的偏差项向量相同，所以将每一时刻求得的不同样本的偏差项
        # 梯度相加求平均值作为该时刻的偏差项梯度
        add_delta = np.nansum(sub_delta_forget, axis=1) / sub_delta_forget.shape[1]
        sub_delta_bias[0] += add_delta.reshape((add_delta.size, 1))
        add_delta = np.nansum(sub_delta_input, axis=1) / sub_delta_input.shape[1]
        sub_delta_bias[1] += add_delta.reshape((add_delta.size, 1))
        add_delta = np.nansum(sub_delta_candidate, axis=1) / sub_delta_candidate.shape[1]
        sub_delta_bias[2] += add_delta.reshape((add_delta.size, 1))
        add_delta = np.nansum(sub_delta_output, axis=1) / sub_delta_output.shape[1]
        sub_delta_bias[3] += add_delta.reshape((add_delta.size, 1))

    # 对权值矩阵和偏差项进行更新
    for k in range(8):
        sub_update_W[k] = sub_W[k] - learning_rate * sub_delta_W[k]
    for k in range(4):
        sub_update_bias[k] = sub_bias[k] - learning_rate * sub_delta_bias[k]

    return sub_update_W, sub_update_bias


# 该函数用于计算剩余的权值矩阵及偏差项的梯度并进行更新
# 该函数输入的参数为：
# 1. sub_c_0为一个字典，长度为m-1,每个键对应的值为除最后一个lstm层外其它层的初始细胞状态矩阵，以Pdf中的例子来看，sub_c_0只含有一个键，
# sub_c_0[0]表示第1个隐藏层的初始化细胞状态矩阵，维数为(20,2382000)
# 2. sub_h_0为一个字典，长度为m-1,每个键对应的值为除最后一个lstm层外其它层的输出矩阵，以Pdf中的例子来看，sub_h_0[0]
# 表示第1个隐藏层的初始化输出矩阵，维数分别为(20,2382000)
# 3. sub_W是一个长度为m的字典，sub_W[k]表示第(k-1)个隐藏层向第k个隐藏层传递时的权值矩阵字典，其元素对应lstm层的8个权值矩阵，顺序与pdf中的顺序相对应
# （第0个隐藏层表示输入层)
# 4. sub_bias是一个字典，长度为m-1,sub_bias[k]表示第(k-1)个隐藏层向第k个隐藏层传递时的偏差项字典，其含有4个元素，对应lstm层的4个偏差项矩阵，顺序与pdf中的顺序相对应
# 5. time_len为该神经网络中用的时间序列长度，例子中即为8
# 6. layer_len为该神经网络中隐藏层的个数
def backward_delta(rootpath, sub_c_0, sub_h_0, sub_W, sub_bias, time_len, layer_len,learning_rate):
    # 与上述函数表示含义相同
    sub_update_W = {}
    sub_delta_W = {}
    sub_update_bias = {}
    sub_delta_bias = {}
    for l in range(layer_len - 1):
        sub_update_W[l] = {}
        sub_delta_W[l] = {}
        sub_update_bias[l] = {}
        sub_delta_bias[l] = {}

    # 对除最后一个隐藏层外其它隐藏层对应的权值矩阵进行遍历
    for l in range(layer_len - 2, -1, -1):

        for i in range(8):
            sub_delta_W[l][i] = 0
        for i in range(4):
            sub_delta_bias[l][i] = 0
        # 与上述函数相同，补充0矩阵
        sub_forget = np.load(rootpath + 'forget_dict\\%d_%d.npy' % (l, 0))
        sub_add_zeros = np.zeros((sub_forget.shape[0], sub_forget.shape[1]))
        np.save(rootpath + 'delta_forget_dict\\%d.npy' % (time_len), sub_add_zeros)
        np.save(rootpath + 'delta_input_dict\\%d.npy' % (time_len), sub_add_zeros)
        np.save(rootpath + 'delta_candidate_dict\\%d.npy' % (time_len), sub_add_zeros)
        np.save(rootpath + 'delta_output_dict\\%d.npy' % (time_len), sub_add_zeros)
        np.save(rootpath + 'forget_dict\\%d_%d.npy' % (l, time_len), sub_add_zeros)
        sub_epsilon_state = sub_add_zeros.copy()
        # 对时间进行逆向遍历
        for i in range(time_len - 1, -1, -1):
            # 后一层的残差
            sub_delta_forget_plus1 = np.load(rootpath + 'delta_forget_dict\\%d.npy' % i)
            sub_delta_input_plus1 = np.load(rootpath + 'delta_input_dict\\%d.npy' % i)
            sub_delta_candidate_plus1 = np.load(rootpath + 'delta_candidate_dict\\%d.npy' % i)
            sub_delta_output_plus1 = np.load(rootpath + 'delta_output_dict\\%d.npy' % i)
            # 后一时刻的残差
            sub_delta_forget = np.load(rootpath + 'delta_forget_dict\\%d.npy' % (i + 1))
            sub_delta_input = np.load(rootpath + 'delta_input_dict\\%d.npy' % (i + 1))
            sub_delta_candidate = np.load(rootpath + 'delta_candidate_dict\\%d.npy' % (i + 1))
            sub_delta_output = np.load(rootpath + 'delta_output_dict\\%d.npy' % (i + 1))
            # 参考pdf中的计算公式
            sub_epsilon_last_output = np.dot(sub_W[l + 1][1].T, sub_delta_forget_plus1) + np.dot(sub_W[l + 1][3].T,
                                                                                                 sub_delta_input_plus1) \
                                      + np.dot(sub_W[l + 1][5].T, sub_delta_candidate_plus1) + np.dot(sub_W[l + 1][7].T,
                                                                                                      sub_delta_output_plus1) \
                                      + np.dot(sub_W[l][0].T, sub_delta_forget) + np.dot(sub_W[l][2].T,
                                                                                         sub_delta_input) + \
                                      np.dot(sub_W[l][4].T, sub_delta_candidate) + np.dot(sub_W[l][6].T, sub_delta_output)

            sub_output = np.load(rootpath + 'output_dict\\%d_%d.npy' % (l, i))
            sub_state = np.load(rootpath + 'state_dict\\%d_%d.npy' % (l, i))
            sub_forget_plus1 = np.load(rootpath + 'forget_dict\\%d_%d.npy' % (l, i + 1))
            sub_candidate = np.load(rootpath + 'candidate_dict\\%d_%d.npy' % (l, i))
            sub_input = np.load(rootpath + 'input_dict\\%d_%d.npy' % (l, i))
            sub_forget = np.load(rootpath + 'forget_dict\\%d_%d.npy' % (l, i))
            # 道理同上述函数
            if (i > 0):
                sub_state_minus1 = np.load(rootpath + 'state_dict\\%d_%d.npy' % (l, i - 1))
            else:
                sub_state_minus1 = sub_c_0[l]

            # 下述过程均与上述函数相同，参考上述注释
            sub_epsilon_state = sub_epsilon_last_output * sub_output * dtanh_vec(sub_state) \
                                + sub_epsilon_state * sub_forget_plus1
            sub_delta_forget = sub_epsilon_state * sub_state_minus1 * drelu_vec(sub_forget)
            np.save(rootpath + 'delta_forget_dict\\%d.npy' % (i), sub_delta_forget)
            sub_delta_input = sub_epsilon_state * sub_candidate * drelu_vec(sub_input)
            np.save(rootpath + 'delta_input_dict\\%d.npy' % (i), sub_delta_input)
            sub_delta_candidate = sub_epsilon_state * sub_input * (1 - sub_candidate * sub_candidate)
            np.save(rootpath + 'delta_candidate_dict\\%d.npy' % (i), sub_delta_candidate)
            sub_delta_output = sub_epsilon_last_output * relu_vec(sub_state) * drelu_vec(sub_output)
            np.save(rootpath + 'delta_output_dict\\%d.npy' % (i), sub_delta_input)

            # 计算delta_W
            if i > 0:
                sub_start = np.load(rootpath + 'start_dict\\%d_%d.npy' % (l + 1, i - 1))
            else:
                sub_start = sub_h_0[l]
            sub_start_minus = np.load(rootpath + 'start_dict\\%d_%d.npy' % (l, i))
            sub_delta_W[l][0] += dot_np(sub_delta_forget, sub_start.T) / sub_start.shape[1]
            sub_delta_W[l][1] += dot_np(sub_delta_forget, sub_start_minus.T) / sub_start_minus.shape[1]
            sub_delta_W[l][2] += dot_np(sub_delta_input, sub_start.T) / sub_start.shape[1]
            sub_delta_W[l][3] += dot_np(sub_delta_input, sub_start_minus.T) / sub_start_minus.shape[1]
            sub_delta_W[l][4] += dot_np(sub_delta_candidate, sub_start.T) / sub_start.shape[1]
            sub_delta_W[l][5] += dot_np(sub_delta_candidate, sub_start_minus.T) / sub_start_minus.shape[1]
            sub_delta_W[l][6] += dot_np(sub_delta_output, sub_start.T) / sub_start.shape[1]
            sub_delta_W[l][7] += dot_np(sub_delta_output, sub_start_minus.T) / sub_start_minus.shape[1]

            add_delta = np.nansum(sub_delta_forget, axis=1) / sub_delta_forget.shape[1]
            sub_delta_bias[l][0] += add_delta.reshape((add_delta.size, 1))
            add_delta = np.nansum(sub_delta_input, axis=1) / sub_delta_input.shape[1]
            sub_delta_bias[l][1] += add_delta.reshape((add_delta.size, 1))
            add_delta = np.nansum(sub_delta_candidate, axis=1) / sub_delta_candidate.shape[1]
            sub_delta_bias[l][2] += add_delta.reshape((add_delta.size, 1))
            add_delta = np.nansum(sub_delta_output, axis=1) / sub_delta_output.shape[1]
            sub_delta_bias[l][3] += add_delta.reshape((add_delta.size, 1))

        for k in range(8):
            sub_update_W[l][k] = sub_W[l][k] - learning_rate * sub_delta_W[l][k]
        for k in range(4):
            sub_update_bias[l][k] = sub_bias[l][k] - learning_rate * sub_delta_bias[l][k]

    return sub_update_W, sub_update_bias