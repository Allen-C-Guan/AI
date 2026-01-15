# 吴恩达 作业ws1 通过回归实现神经网络
import numpy as np
import h5py
    
# 配套库   
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
	
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
%matplotlib inline

# 读取训练数据
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
assert train_set_y_orig.shape == (1, 209)
plt.imshow(test_set_x_orig[5])

# 高维矩阵向量化（扁平化）
train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
train_set_y_orig  = train_set_y_orig.reshape(train_set_y_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
test_set_y_orig  = test_set_y_orig.reshape(test_set_y_orig.shape[0], -1).T

# 校验shape
print(train_set_x_flatten.shape)
print(train_set_y_orig.shape)
print(test_set_x_flatten.shape)
print(test_set_y_orig.shape)


# 训练数据集标准化
train_set_x_flatten_norm = train_set_x_flatten/255
test_set_x_flatten_norm = test_set_x_flatten/255


# 定义所需数学计算
def sigmoid(z):
    return 1/(1+np.exp(-z))

# 传播：
# input：w, b, x_train, y.train
# output: dw,db,J
# 1. 先正向计算Y_predict
# 2. 利用反向传播导数公式，求解dw和db
# 3. 可以计算cost函数观察 cost趋势
def propagate(w, b, x_flat, y):
    w  = w.reshape(w.shape[0],1) # 这里是个大坑，1维np array的减法有点奇怪
    m = x_flat.shape[1]
    a = sigmoid(np.dot(w.T,x_flat)+b) #a dim (1,sample_num)
    J = (-1/m) * (np.sum(y * np.log(a) + (1-y) * np.log(1-a)))
    dw = (1/m) * np.dot(x_flat, (a - y).T) # dw dim (num, 1)
    db = (1/m) * np.sum(a - y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    return dw, db,J

# 训练
# input: w,b,X_train, Y_train, 迭代次数，学习率
# output：训练好的w和b
# 通过迭代调用正向和反向传播，得到dw和db
# 更新参数，w -= w - dw * step_size
def optimize(w, b, X, Y, num_iter, learning_rate, printCost = False):
    cost_list = []
    for i in range(num_iter):
        dw, db,J = propagate(w,b,X,Y)
        cost_list.append(J)
        if printCost and i%100 == 0:
            print("cost", J)
        w = w.reshape(w.shape[0],1) # 这里是个大坑，1维np array的减法有点奇怪
        w = w - learning_rate * dw
        b = b - learning_rate * db
        assert(dw.shape == w.shape)
        assert(db.dtype == float)
    paras = {"w":w, "b":b}
    grads = {"dw":dw, "db":db}
    return paras, grads,cost_list
# 预测
# input: 输入好的w，b，x_predict
# output: 预测结果
def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        A[0, i] = 1 if A[0,i] > 0.5 else 0
    return A
	
# 模型
# 1. 先训练得到 w和b
# 2. 再用w和b，输入x_test, 预测y_test
	
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w = np.zeros(X_train.shape[0])
    b = 0
    paras, grads,cost_list = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_prediction_train = predict(paras["w"], paras["b"],X_train)
    Y_prediction = predict(paras["w"], paras["b"],X_test)
    
    sample_num_train = Y_prediction_train.shape[1]
    sample_num_test = Y_test.shape[1]
    accuary_train_rate = 1- np.sum(np.abs(Y_prediction_train - Y_train))/sample_num_train
    accuary_test_rate = 1- np.sum(np.abs(Y_prediction - Y_test))/sample_num_test
    print(accuary_train_rate)
    print(accuary_test_rate)
    return accuary_train_rate, accuary_test_rate, cost_list
    
accuary_train_rate, accuary_test_rate, cost_list = model(train_set_x_flatten_norm, train_set_y_orig, test_set_x_flatten_norm, test_set_y_orig, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


plt.plot(cost_list)
plt.figure(figsize = (1000,4))
plt.show()


# 坑b的1维 np array 
a = np.array([[1,2,3,4],[5,6,7,8]]).T
b = np.array([2,4])
print(b.shape)
print(a.shape)
print("b", b)
print(np.dot(a,b)) #这不报错，说明b可以是列向量
print(np.dot(a, b.T)) #这不报错，说明b也可以是行向量

# 他到底是行还是列，没个b数，所以尽量不要用1维向量
c = np.array([[2,4]]) #使用二维向量
print(c.shape)
