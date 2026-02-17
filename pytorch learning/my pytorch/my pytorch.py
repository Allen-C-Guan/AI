import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = data                 # 实际数值（numpy 数组或标量）
        self.requires_grad = requires_grad
        self.grad = None                  # 梯度累积
        self.grad_fn = grad_fn             # 指向生成此 Tensor 的 Function

    def backward(self, gradient=None):
        # 根节点初始化
        if gradient is None:
            # 如果输出是标量，初始梯度为 1
            gradient = np.array(1.0)
        # tenor反向传播中，继续向左call operator
        if self.grad_fn is not None:
            self.grad_fn.backward(self, gradient)

        # 梯度保存
        self.grad = gradient if self.grad is None else self.grad + gradient

class Operator:
    def __call__(self, *inputs):
        '''

        Args:
            *inputs: 被operator操作的tensor的集合

        Returns:
            正向传播的结果
        '''
        # 多态执行前向计算的同时记录被操作的tensor
        outputs = self.forward(*inputs)
        # 标记需要梯度的输入
        requires_grad = any(i.requires_grad for i in inputs)
        if requires_grad:
            # 输出的结果中，将自己绑定在output tensor中
            outputs = [Tensor(out, requires_grad=True, grad_fn=self) for out in [outputs]]
            '''
            例如这是mul x * y，则运算记录了如下操作：
            1. 记录了 x 和 y是 mul的操作对象
            2. output 是 mult操作生成的
            '''
        return outputs if len(outputs) > 1 else outputs[0]  #这里为啥要这么写？

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, output, grad_output):
        raise NotImplementedError

class Add(Operator):
    def forward(self, t1, t2):
        self.t1 = t1  # 保存原始 Tensor（引用）
        self.t2 = t2
        return t1.data + t2.data

    def backward(self, output, grad_output):
        # 将自己的梯度累计上后，持续递归向左深度探索node
        if self.t1.requires_grad:
            self.t1.backward(grad_output)
        if self.t2.requires_grad:
            self.t2.backward(grad_output)

class Mul(Operator):
    def forward(self, t1, t2):
        self.t1 = t1
        self.t2 = t2
        return t1.data * t2.data

    def backward(self, output, grad_output):
        # 乘法局部梯度：∂out/∂a = b, ∂out/∂b = a
        if self.t1.requires_grad:
            self.t1.backward(grad_output * self.t2.data)
        if self.t2.requires_grad:
            self.t2.backward(grad_output * self.t1.data)

def __add__(self, other):
    return Add()(self, other)

def __mul__(self, other):
    return Mul()(self, other)

Tensor.__add__ = __add__
Tensor.__mul__ = __mul__

if __name__ == "__main__" :
    # 创建需要梯度的叶子张量
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    w = Tensor(1.0, requires_grad=True)

    # 前向计算
    t = x * y        # 记录了t.grad_fn = Mul， 而Mul操作的saved_tensors为x和y
    z = t + w        # 记录了z.grad_fn = Add， 而Add操作的save tensor为t和w
    
    '''
    整个计算的过程就是建立图的过程。
    上述例子中
    t指向了mul，而mul指向了x和y
    z指向了add，而add又指向了t和w
    因此形成了一个如下的图：
    x
     \
      mul -- t
     /         \
    y           add -- z
               /
             w 
    因此在在forward流程中（从左向右建立图的过程）
    
    operator需要实现：
    1. input tensor到operator的绑定：forward：返回计算结果，并将所有的operator的功能就是记录被操作的tensor，t1和t2 
    2. output tensor到operator的绑定：operator.__call__中，将operator实例本身绑定在output tensor中，并返回output
    
    在backward过程中（从右向左，深度遍历，每次路过一个operator，累计梯度将gradient = self.grad * gradient)
    1. operator 负责在gradient上加上自己的梯度，所得结果再向左，逐个传给所有指向自己的tensor，实现深度遍历
    2. tensor
        1).保存累计到目前节点的 gradient到self.grad上
        2).负责向左call操作自己的operator，继续向更深处遍历
    '''

    # 反向传播
    z.backward()     # 初始梯度为 1.0

    # 查看梯度
    print(x.grad)    # 应输出 3.0  (∂z/∂x = y)
    print(y.grad)    # 应输出 2.0  (∂z/∂y = x)
    print(w.grad)    # 应输出 1.0  (∂z/∂w = 1)
    print(t.grad)