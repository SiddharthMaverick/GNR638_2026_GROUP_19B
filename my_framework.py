import ctypes
import os
import random
import math
import time

_GLOBAL_SEED_COUNTER = 42

def manual_seed(seed):
    """Sets the starting seed for weight initialization"""
    global _GLOBAL_SEED_COUNTER
    _GLOBAL_SEED_COUNTER = seed
    
def _get_next_seed():
    """Returns a unique seed for a layer and increments the counter"""
    global _GLOBAL_SEED_COUNTER
    current = _GLOBAL_SEED_COUNTER
    _GLOBAL_SEED_COUNTER += 1
    return current

# --- 1. Load Backend ---
lib_path = os.path.abspath("libbackend.dll")
if not os.path.exists(lib_path):
    raise FileNotFoundError("Compile backend.cpp first!")

lib = ctypes.CDLL(lib_path)

# Define generic pointer types
F_PTR = ctypes.POINTER(ctypes.c_float)
I_PTR = ctypes.POINTER(ctypes.c_int)


# cross_entropy_loss
lib.cross_entropy_loss.argtypes = [F_PTR, F_PTR, F_PTR, ctypes.c_int, ctypes.c_int]
lib.cross_entropy_loss.restype = ctypes.c_float  # <--- THIS FIXES THE NEGATIVE LOSS

# sgd_step
lib.sgd_step.argtypes = [F_PTR, F_PTR, ctypes.c_int, ctypes.c_float]

# linear_forward / backward
lib.linear_forward.argtypes = [F_PTR, F_PTR, F_PTR, F_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.linear_backward.argtypes = [F_PTR, F_PTR, F_PTR, F_PTR, F_PTR, F_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int]

# conv2d_forward / backward
lib.conv2d_forward.argtypes = [F_PTR, F_PTR, F_PTR, F_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.conv2d_backward.argtypes = [F_PTR, F_PTR, F_PTR, F_PTR, F_PTR, F_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

# maxpool / relu
lib.maxpool_forward.argtypes = [F_PTR, F_PTR, I_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.maxpool_backward.argtypes = [F_PTR, F_PTR, I_PTR, ctypes.c_int]
lib.relu_forward.argtypes = [F_PTR, F_PTR, ctypes.c_int]
lib.relu_backward.argtypes = [F_PTR, F_PTR, F_PTR, ctypes.c_int]

# init_weights
lib.init_weights.argtypes = [F_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int]


# --- 2. Tensor Class (No NumPy!) ---
class Tensor:
    def __init__(self, shape, data=None, requires_grad=False):
        self.shape = shape
        self.size = 1
        for s in shape: self.size *= s
        self.requires_grad = requires_grad
        self.grad = None
        
        # Allocate memory using ctypes array (Standard Library)
        self.array_type = ctypes.c_float * self.size
        
        if data is None:
            self.data = self.array_type() # Zero init
        elif isinstance(data, (list, tuple)):
            # Flatten list if needed (basic flattening)
            flat = []
            def _flatten(l):
                for item in l:
                    if isinstance(item, (list, tuple)): _flatten(item)
                    else: flat.append(item)
            _flatten(data)
            self.data = self.array_type(*flat)
        else:
            self.data = data # Assume it's already a c_array

        self.ptr = ctypes.cast(self.data, F_PTR)

    def zero_grad(self):
        if self.grad is None:
            self.grad = Tensor(self.shape)
        else:
            ctypes.memset(self.grad.data, 0, ctypes.sizeof(self.grad.data))

    def flatten(self):
        return Tensor((self.shape[0], -1), data=self.data, requires_grad=self.requires_grad)

# --- 3. Layers ---

class Module:
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, x): return self.forward(x)
    def parameters(self): return []

class Linear(Module):
    def __init__(self, in_feat, out_feat):
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        # Weights & Bias
        self.weights = Tensor((in_feat, out_feat), requires_grad=True)
        self.bias = Tensor((out_feat,), requires_grad=True)
        
        layer_seed = _get_next_seed()
        # Init weights (He)
        lib.init_weights(self.weights.data, self.weights.size, self.in_feat, self.out_feat, layer_seed)
        
        self.cache = None

    def forward(self, x):
        batch = x.shape[0]
        out = Tensor((batch, self.out_feat), requires_grad=True)
        
        # Call C++
        lib.linear_forward(x.ptr, self.weights.ptr, self.bias.ptr, out.ptr, batch, self.in_feat, self.out_feat)
        
        self.cache = x
        return out

    def backward(self, dout):
        x = self.cache
        batch = x.shape[0]
        dx = Tensor(x.shape)
        
        # Alloc grads if not exists
        if self.weights.grad is None: self.weights.zero_grad()
        if self.bias.grad is None: self.bias.zero_grad()
        
        lib.linear_backward(x.ptr, self.weights.ptr, dout.ptr, dx.ptr,
                            self.weights.grad.ptr, self.bias.grad.ptr,
                            batch, self.in_feat, self.out_feat)
        return dx

    def parameters(self): return [self.weights, self.bias]

class Conv2d(Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0):
        self.in_c = in_c
        self.out_c = out_c
        self.k = kernel_size
        self.s = stride
        self.p = padding
        
        self.weights = Tensor((out_c, in_c, kernel_size, kernel_size), requires_grad=True)
        self.bias = Tensor((out_c,), requires_grad=True)
        
        layer_seed = _get_next_seed()
        
        lib.init_weights(self.weights.ptr, self.weights.size, in_c*kernel_size*kernel_size, out_c,layer_seed)
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        H_out = (H + 2*self.p - self.k) // self.s + 1
        W_out = (W + 2*self.p - self.k) // self.s + 1
        
        out = Tensor((N, self.out_c, H_out, W_out), requires_grad=True)
        
        lib.conv2d_forward(x.ptr, self.weights.ptr, self.bias.ptr, out.ptr,
                           N, C, H, W, self.out_c, self.k, self.s, self.p)
        self.cache = x
        return out

    def backward(self, dout):
        N, C, H, W = self.cache.shape
        dx = Tensor(self.cache.shape)
        
        # Alloc grads if not exists
        if self.weights.grad is None: self.weights.zero_grad()
        if self.bias.grad is None: self.bias.zero_grad()
        
        lib.conv2d_backward(self.cache.ptr, self.weights.ptr, dout.ptr, dx.ptr,
                            self.weights.grad.ptr, self.bias.grad.ptr,
                            N, C, H, W, self.out_c, self.k, self.s, self.p)
        return dx


    def parameters(self): return [self.weights, self.bias]

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride):
        self.k = kernel_size
        self.s = stride
        self.mask = None
        self.input_shape = None

    def forward(self, x):
        N, C, H, W = x.shape
        H_out = (H - self.k) // self.s + 1
        W_out = (W - self.k) // self.s + 1
        
        out = Tensor((N, C, H_out, W_out))
        self.mask = (ctypes.c_int * out.size)() # Alloc mask array
        self.input_shape = x.shape
        
        lib.maxpool_forward(x.ptr, out.ptr, self.mask, N, C, H, W, self.k, self.s)
        return out

    def backward(self, dout):
        dx = Tensor(self.input_shape)
        lib.maxpool_backward(dout.ptr, dx.ptr, self.mask, dout.size)
        return dx

class ReLU(Module):
    def __init__(self): self.cache = None
    
    def forward(self, x):
        out = Tensor(x.shape)
        lib.relu_forward(x.ptr, out.ptr, x.size)
        self.cache = x
        return out

    def backward(self, dout):
        dx = Tensor(self.cache.shape)
        lib.relu_backward(self.cache.ptr, dout.ptr, dx.ptr, dout.size)
        return dx

# --- 4. Optimizer ---
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                lib.sgd_step(p.ptr, p.grad.ptr, p.size, ctypes.c_float(self.lr))

    def zero_grad(self):
        for p in self.params: p.zero_grad()

# --- 5. Loss ---
class CrossEntropyLoss:
    def __call__(self, logits, targets):
        batch = logits.shape[0]
        classes = logits.shape[1]
        
        # Targets assumed one-hot tensor
        grad = Tensor(logits.shape)
        
        # Zero-initialize gradient before passing to C++ (prevent uninitialized memory)
        ctypes.memset(grad.data, 0, ctypes.sizeof(grad.array_type))
        
        # Returns float loss, writes gradient into `grad`
        loss_val = lib.cross_entropy_loss(logits.ptr, targets.ptr, grad.ptr, batch, classes)
        return loss_val, grad