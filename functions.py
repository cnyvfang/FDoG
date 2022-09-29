from utils import *
import numpy as np
import cv2 as cv

def detect_edge(img, flow, thresh, sigma_c, rho,
                sigma_m, tau):
    """
    Detect edge on input image based of edge tangent flow
    基于边缘切线流检测输入图像的边缘
    """
    h, w = img.shape
    # normalize input image
    img = img / 255.0

    # create two gauss filter
    gauss_c = make_gauss_filter(sigma_c, threshold=thresh)
    gauss_s = make_gauss_filter(sigma_c * 1.6,
                                threshold=thresh)

    # shrink filter to the same size
    w_gauss_c, w_gauss_s = len(gauss_c) // 2, len(gauss_s) // 2
    w_fdog = min(w_gauss_c, w_gauss_s)
    gauss_c = shrink_array(gauss_c, w_gauss_c, w_fdog)
    gauss_s = shrink_array(gauss_s, w_gauss_s, w_fdog)

    # do padding because some vectorized operations
    # may accross the boundary of image
    img = np.pad(img,
                 ((w_fdog, w_fdog), (w_fdog, w_fdog)))

    # start coords of each pixel (shifted by width of filter)
    sx, sy = np.meshgrid(np.arange(w), np.arange(h))
    start = np.concatenate((sx[np.newaxis, ...],
                            sy[np.newaxis, ...]), axis=0) + w_fdog

    # steps of each pixel
    steps = np.arange(-w_fdog, w_fdog + 1).reshape(
        -1, 1, 1, 1)
    steps = np.repeat(steps, repeats=2, axis=1)

    # rotate flow to get gradient
    grad = np.empty_like(flow)
    grad[0, ...] = flow[1, ...]
    grad[1, ...] = -flow[0, ...]

    # take steps along the gradient
    xy = start + (steps * grad)
    ixy = np.round(xy).astype('int32')
    ix, iy = np.split(ixy, indices_or_sections=2, axis=1)
    ix = ix.reshape(2 * w_fdog + 1, h, w)
    iy = iy.reshape(2 * w_fdog + 1, h, w)

    # neighbors of each pixel along the gradient
    neighbors = img[iy, ix]

    # apply dog filter in gradient's direction
    gauss_c = gauss_c.reshape(2 * w_fdog + 1, 1, 1)
    img_gauss_c = np.sum(gauss_c * neighbors, axis=0) \
                  / np.sum(gauss_c)

    gauss_s = gauss_s.reshape(2 * w_fdog + 1, 1, 1)
    img_gauss_s = np.sum(gauss_s * neighbors, axis=0) \
                  / np.sum(gauss_s)
    img_fdog = img_gauss_c - rho * img_gauss_s

    # remove those pixels with zero gradient
    zero_grad_mask = np.logical_and(
        grad[0, ...] == 0, grad[1, ...] == 0)
    img_fdog[zero_grad_mask] = np.max(img_fdog)

    # make gauss filter along tangent vector
    gauss_m = make_gauss_filter(sigma_m)
    w_gauss_m = len(gauss_m) // 2

    # initialize with a negative weight for coding's convenience
    edge = -gauss_m[w_gauss_m] * img_fdog
    weight_acc = np.full_like(img_fdog,
                              fill_value=-gauss_m[w_gauss_m])

    # do padding
    img_fdog = np.pad(img_fdog,
                      ((w_gauss_m, w_gauss_m), (w_gauss_m, w_gauss_m)))
    zero_grad_mask = np.pad(zero_grad_mask,
                            ((w_gauss_m, w_gauss_m), (w_gauss_m, w_gauss_m)))
    flow = np.pad(flow, ((0, 0),
                         (w_gauss_m, w_gauss_m), (w_gauss_m, w_gauss_m)))

    # start coords of each pixcel
    sx, sy = np.meshgrid(np.arange(w), np.arange(h))
    sx += w_gauss_m; sy += w_gauss_m

    # forward mask, indicate whether a pixel need to keep
    # going along tangent vector or not
    forward_mask = np.full(shape=(h, w), fill_value=True,
                           dtype='bool')

    # convert dtype from integer to float for accumulating
    # steps along tangent vector
    x = sx.astype('float32')
    y = sy.astype('float32')
    ix, iy = np.round(x).astype('int32'), \
             np.round(y).astype('int32')

    # start
    for i in range(w_gauss_m + 1):
        # get neighbors of each pixel w.r.t its coordinate
        neighbors = img_fdog[iy, ix]

        # multiply weight, ignore those pixels who stopped
        weight = gauss_m[w_gauss_m + i]
        edge[forward_mask] += (neighbors * weight)[forward_mask]
        weight_acc[forward_mask] += weight

        # take a step along tangent vector w.r.t coordinate
        x += flow[0, iy, ix]
        y += flow[1, iy, ix]

        # update coordinates
        ix, iy = np.round(x).astype('int32'), \
                 np.round(y).astype('int32')

        # update each pixels' status
        none_zero_mask = np.logical_not(
            zero_grad_mask[iy, ix])
        forward_mask = np.logical_and(
            forward_mask, none_zero_mask)

    # going along the reversed tangent vector
    forward_mask = np.full(shape=(h, w), fill_value=True,
                           dtype='bool')
    x = sx.astype('float32')
    y = sy.astype('float32')
    ix, iy = np.round(x).astype('int32'), np.round(y).astype('int32')

    for i in range(w_gauss_m + 1):
        neighbor = img_fdog[iy, ix]

        weight = gauss_m[w_gauss_m - i]
        edge[forward_mask] += (neighbor * weight)[forward_mask]
        weight_acc[forward_mask] += weight

        # take a step
        x -= flow[0, iy, ix]
        y -= flow[1, iy, ix]
        ix, iy = np.round(x).astype('int32'), np.round(y).astype('int32')

        none_zero_mask = np.logical_not(
            zero_grad_mask[iy, ix])
        forward_mask = np.logical_and(
            forward_mask, none_zero_mask)

    # postprocess
    edge /= weight_acc
    edge[edge > 0] = 1
    edge[edge <= 0] = 1 + np.tanh(edge[edge <= 0])
    edge = (edge - np.min(edge)) /  (np.max(edge) - np.min(edge))

    # binarize
    edge[edge < tau] = 0
    edge[edge >= tau] = 255
    return edge.astype('uint8')

