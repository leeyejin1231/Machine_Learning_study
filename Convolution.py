from collections import OrderedDict
import numpy as np

class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        # if isinstance(kernel_size, int):
        #     kernel_size = (kernel_size, kernel_size)

        self.w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = np.zeros(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        batch_size, in_channel, _, _ = x.shape
        conv = self.convolution(x, self.w, self.b, self.stride, self.pad)
        self.output_shape = conv.shape
        return conv

    def convolution(self, x, w, b, stride=1, pad=0):        
        # Check validity
        check_conv_validity(x, w, stride, pad)

        if pad > 0:
            x = zero_pad(x, pad)
        
        self.x = x

        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        Hp = H - HH +1
        Wp = W - WW +1
        conv = []
        pre_conv = []
        
        pre_x = x[0]
        
        for f in range(F):
            pre_w = w[f]
            conv2 = []
            for h in range(Hp):
                v_start = h
                v_end = h+HH
                conv1 = []
                for ww in range(Wp):
                    h_start = ww
                    h_end = ww+WW
                    slice_x = pre_x[:, v_start:v_end, h_start:h_end]
                    conv3 = []
                    for k in range(C):
                        a_x = slice_x[k]
                        a_w = pre_w[k]
                        conv3.append(sum(sum(np.multiply(a_x, a_w))))
                    conv1.append(sum(conv3))
                conv2.append(conv1)
            pre_conv.append(conv2)

        conv.append(pre_conv)

        conv = np.array(conv)
        
        return conv
