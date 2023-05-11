from giagrad.nn.layers.convs.convnd import ConvND

class Conv2D(ConvND):
    __valid_input_dims = [2, 3]
    __error_message  = "Conv2D only accepts input tensors of shapes:\n"
    __error_message += "(N, C_in, H_in, W_in) or (C_in, H_in, W_in)\n"
    __error_message += "got: {shape}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
