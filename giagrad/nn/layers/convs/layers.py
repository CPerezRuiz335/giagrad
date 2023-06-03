from giagrad.nn.layers.convs.convnd import ConvND

class Conv1D(ConvND):
    __valid_input_dims = [2, 3]
    __error_message  = "Conv2D only accepts input tensors of shapes:\n"
    __error_message += "(N, C_in, W_in) or (C_in, W_in)\n"
    __error_message += "got: {shape}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Conv2D(ConvND):
    __valid_input_dims = [3, 4]
    __error_message  = "Conv2D only accepts input tensors of shapes:\n"
    __error_message += "(N, C_in, H_in, W_in) or (C_in, H_in, W_in)\n"
    __error_message += "got: {shape}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Conv3D(ConvND):
    __valid_input_dims = [4, 5]
    __error_message  = "Conv2D only accepts input tensors of shapes:\n"
    __error_message += "(N, C_in, D_in, H_in, W_in) or (C_in, D_in, H_in, W_in)\n"
    __error_message += "got: {shape}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
