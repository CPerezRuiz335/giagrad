from typing import Optional, Tuple, Union
import numpy as np
from numbers import Integral
from numpy.lib.stride_tricks import as_strided


def sliding_window_view(arr, window_shape, step, dilation=None):
    if not hasattr(window_shape, "__iter__"):
        raise TypeError(
            f"`window_shape` must be a sequence of positive integers, got: {window_shape}"
        )
    window_shape = tuple(window_shape)
    if not all(isinstance(i, Integral) and i > 0 for i in window_shape):
        raise TypeError(
            f"`window_shape` must be a sequence of positive integers, "
            f"got: {window_shape}"
        )

    if len(window_shape) > arr.ndim:
        raise ValueError(
            f"`window_shape` ({window_shape}) cannot specify more values than "
            f"`arr.ndim` ({arr.ndim})."
        )

    if not isinstance(step, Integral) and not hasattr(step, "__iter__"):
        raise TypeError(
            f"`step` must be a positive integer or a sequence of positive "
            f"integers, got: {step}"
        )

    step = (
        (int(step),) * len(window_shape) if isinstance(step, Integral) else tuple(step)
    )

    if not all(isinstance(i, Integral) and i > 0 for i in step):
        raise ValueError(
            f"`step` must be a positive integer or a sequence of positive "
            f"integers, got: {step}"
        )

    if any(i > j for i, j in zip(window_shape[::-1], arr.shape[::-1])):
        raise ValueError(
            f"Each size of the window-shape must fit within the trailing "
            f"dimensions of `arr`."
            f"{window_shape} does not fit in {arr.shape[-len(window_shape) :]}"
        )

    if (
        dilation is not None
        and not isinstance(dilation, Integral)
        and not hasattr(dilation, "__iter__")
    ):
        raise TypeError(
            f"`dilation` must be None, a positive integer, or a sequence of "
            f"positive integers, got: {dilation}"
        )
    if dilation is None:
        dilation = np.ones((len(window_shape),), dtype=int)
    else:
        if isinstance(dilation, Integral):
            dilation = np.full((len(window_shape),), fill_value=dilation, dtype=int)
        else:
            np.asarray(dilation)

        if not all(isinstance(i, Integral) and i > 0 for i in dilation) or len(
            dilation
        ) != len(window_shape):
            raise ValueError(
                f"`dilation` must be None, a positive integer, or a sequence of "
                f"positive integers with the same length as `window_shape` "
                f"({window_shape}), got: {dilation}"
            )
        if any(
            w * d > s
            for w, d, s in zip(window_shape[::-1], dilation[::-1], arr.shape[::-1])
        ):
            raise ValueError(
                f"The dilated window ({tuple(w * d for w, d in zip(window_shape, dilation))}) "
                f"must fit within the trailing "
                f"dimensions of `arr` ({arr.shape[-len(window_shape) :]})"
            )

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    step = np.array(step)  # (Sx, ..., Sz)
    window_shape = np.array(window_shape)  # (Wx, ..., Wz)
    in_shape = np.array(arr.shape[-len(step) :])  # (x, ... , z)
    nbyte = arr.strides[-1]  # size, in bytes, of element in `arr`

    # per-byte strides required to fill a window
    win_stride = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)

    # per-byte strides required to advance the window
    step_stride = tuple(win_stride[-len(step) :] * step)

    # update win_stride to accommodate dilation
    win_stride = np.array(win_stride)
    win_stride[-len(step) :] *= dilation
    win_stride = tuple(win_stride)

    # tuple of bytes to step to traverse corresponding dimensions of view
    # see: 'internal memory layout of an ndarray'
    stride = tuple(int(nbyte * i) for i in step_stride + win_stride)

    # number of window placements along x-dim: X = (x - (Wx - 1)*Dx + 1) // Sx + 1
    out_shape = tuple((in_shape - ((window_shape - 1) * dilation + 1)) // step + 1)

    # ([X, (...), Z], ..., [Wx, (...), Wz])
    out_shape = out_shape + arr.shape[: -len(step)] + tuple(window_shape)
    out_shape = tuple(int(i) for i in out_shape)

    return as_strided(arr, shape=out_shape, strides=stride, writeable=False)



class ConvND:
    def __call__(self, x, w, *, stride, padding=0, dilation=1):
        self.variables = (x, w)
        # x ... data:    (N, C, X0, X1, ...)
        # w ... filters: (F, C, W0, W1, ...)

        x = x
        w = w

        assert x.ndim > 2
        assert x.ndim == w.ndim
        assert (
            w.shape[1] == x.shape[1]
        ), "The channel-depth of the batch and filters must agree"

        num_conv_channels = w.ndim - 2
        x_shape = np.array(
            x.shape[2:]
        )  # (X0, ...): shape of the channels being convolved over
        w_shape = np.array(w.shape[2:])  # (W0, ...): shape of each conv filter

        dilation = (
            np.array((dilation,) * num_conv_channels)
            if isinstance(dilation, Integral)
            else np.array(dilation, dtype=int)
        )

        assert len(dilation) == num_conv_channels and all(
            d >= 1 and isinstance(d, Integral) for d in dilation
        )

        padding = (
            np.array((padding,) * num_conv_channels)
            if isinstance(padding, Integral)
            else np.array(padding, dtype=int)
        )
        assert len(padding) == num_conv_channels and all(
            p >= 0 and isinstance(p, Integral) for p in padding
        )

        stride = (
            np.array((stride,) * num_conv_channels)
            if isinstance(stride, Integral)
            else np.asarray(stride, dtype=int)
        )
        assert len(stride) == num_conv_channels and all(
            s >= 1 and isinstance(s, Integral) for s in stride
        )

        out_shape = (
            x_shape + 2 * padding - ((w_shape - 1) * dilation + 1)
        ) / stride + 1

        if not all(i.is_integer() and i > 0 for i in out_shape):
            msg = "Stride and kernel dimensions are incompatible: \n"
            msg += f"Input dimensions: {tuple(x_shape)}\n"
            msg += f"Stride dimensions: {tuple(stride)}\n"
            msg += f"Kernel dimensions: {tuple(w_shape)}\n"
            msg += f"Padding dimensions: {tuple(padding)}\n"
            msg += f"Dilation dimensions: {tuple(dilation)}\n"
            raise ValueError(msg)

        self.padding = padding
        self.stride = stride
        self.dilation = dilation

        # symmetric 0-padding for X0, X1, ... dimensions
        axis_pad = tuple((i, i) for i in (0, 0, *padding))
        x = np.pad(x, axis_pad, mode="constant") if sum(padding) else x

        # (G0, ...) is the tuple of grid-positions for placing each window (not including stride)
        # (N, C, X0, ...) -> (G0, ..., N, C, W0, ...)
        windowed_data = sliding_window_view(
            x, window_shape=w_shape, step=self.stride, dilation=self.dilation
        )

        w_conv_channels = list(range(1, num_conv_channels + 2))  # C, W0, ...
        window_conv_channels = [
            i + 1 + num_conv_channels  # C, W0, ...
            for i in range(num_conv_channels + 1)
        ]

        # (F, C, W0, ...) ⋆ (G0, ..., N, C, W0, ...) -> (F, G0, ..., N)
        conv_out = np.tensordot(
            w, windowed_data, axes=[w_conv_channels, window_conv_channels]
        )

        # (F, G0, ..., N) -> (N, F, G0, ...)
        out = np.moveaxis(conv_out, source=-1, destination=0)
        return out if out.flags["C_CONTIGUOUS"] else np.ascontiguousarray(out)

    def backward_var(self, grad, index, **kwargs):
        x, w = (i.data for i in self.variables)
        num_conv_channels = grad.ndim - 2

        if index == 0:  # backprop through x
            x_shape = x.shape[:2] + tuple(
                i + 2 * p for i, p in zip(x.shape[-num_conv_channels:], self.padding)
            )
            dx = np.zeros(x_shape, dtype=x.dtype)  # (N, C, X0, ...)

            # `gp` stores all of the various broadcast multiplications of each grad
            # element against the conv filter.
            # (N, F, G0, ...) -tdot- (F, C, W0, ...) --> (N, G0, ..., C, W0, ...)
            gp = np.tensordot(grad, w, axes=[[1], [0]])
            for ind in np.ndindex(grad.shape[-num_conv_channels:]):
                # ind: (g0, ...) - grid-position of filter placement
                slices = tuple(
                    slice(i * s, i * s + w * d, d)
                    for i, w, s, d in zip(ind, w.shape[2:], self.stride, self.dilation)
                )
                # Add (grad-element * filter) to each appropriate window position in `dx`
                # dx[N, C, g0*s0 : g0*s0 + w0*d0 : d0, (...)] += gp[N, g0, (...), C, W0, (...)]
                dx[(..., *slices)] += gp[(slice(None), *ind, ...)]

            # remove padding from dx
            if sum(self.padding):
                no_pads = tuple(slice(p, -p if p else None) for p in self.padding)
                dx = dx[(..., *no_pads)]
            return dx

        else:  # backprop through w
            # backprop into f
            # symmetric 0-padding for H, W dimensions
            axis_pad = tuple((i, i) for i in (0, 0, *self.padding))
            x = np.pad(x, axis_pad, mode="constant") if sum(self.padding) else x

            # (G0, ...) is the tuple of grid-indices for placing each window (not including stride)
            # (N, C, X0, ...) -> (G0, ..., N, C, W0, ...)
            windowed_data = sliding_window_view(
                x, window_shape=w.shape[2:], step=self.stride, dilation=self.dilation
            )

            # (N, F, G0, ...) -tdot- (G0, ..., N, C, W0, ...) --> (F, C, W0, ...)
            grad_axes = list(range(2, num_conv_channels + 2)) + [0]  # (G0, ..., N)
            window_axes = list(range(num_conv_channels + 1))  # (G0, ..., N)
            return np.tensordot(grad, windowed_data, axes=[grad_axes, window_axes])
