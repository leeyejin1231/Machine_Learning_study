class MaxPoolingLayer:
    def __init__(self, kernel_size, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        ##############################################################################################
        # Max-Pooling Layer Forward. Pool maximum value by striding W.
        # 
        # [Input]
        # x: 4-D input batch data
        # - Shape : (Batch size, In Channel, Height, Width)
        # 
        # [Output]
        # max_pool : max_pool result
        # - Shape : (Batch size, Out Channel, Pool_Height, Pool_Width)
        # - Pool_Height & Pool_Width can be calculated using 'Height', 'Width', 'Kernel_size', 'Stride'
        ###############################################################################################
        max_pool = None
        N, C, H, W = x.shape
        check_pool_validity(x, self.kernel_size, self.stride)
        
        self.x = x
        
        max_pool = []
        for n in range(N):
          pre_pool = []
          for c in range(C):
            pre_pool2 = []
            target_layer = self.x[n][c]
            for h in range(H - self.kernel_size + 1):
              dummy_row = []
              for ww in range(W - self.kernel_size + 1):
                filtered_layer = (target_layer[h:h+self.kernel_size, ww:ww+self.kernel_size])
                dummy_row.append(max(map(max, filtered_layer)))
              pre_pool2.append(dummy_row)
            pre_pool.append(pre_pool2)
          max_pool.append(pre_pool)
          max_pool = np.array(max_pool)
        self.output_shape = max_pool.shape
        return max_pool
