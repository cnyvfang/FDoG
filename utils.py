import numpy as np
import cv2 as cv

def cv_imshow(img, title='[TEST]', wait=0.5, move=None):
    """ Wrapper for cv.imshow(...) """
    cv.imshow(title, img)
    if move:
        cv.moveWindow(title, *move)
    return cv.waitKey(int(wait * 1000))


def find_neighbors(x, ksize, s, out_h, out_w):
    """ Get sliding windows using numpy's stride tricks. """
    in_c, in_h, in_w = x.shape
    shape = (out_h, out_w, in_c, ksize, ksize)
    itemsize = x.itemsize
    strides = (
        s    * in_w * itemsize,
        s    * itemsize,
        in_w * in_h * itemsize,
        in_w * itemsize,
        itemsize
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape,
        strides=strides)

def initialze_flow(img, sobel_size):
    """ Initialize edge tangent flow, contains the
    following steps:
        (1) normalize input image
        (2) compute gradient using sobel operator
        (3) compute gradient magnitude
        (4) normalize gradient and magnitude
        (5) rotate gradient to get tanget vector

        (1) 归一化输入图像
        (2) 使用索贝尔算子计算梯度
        (3) 计算梯度的大小
        (4) 将梯度和幅度归一化
        (5) 旋转梯度以获得切线矢量
    """
    img = cv.normalize(img, dst=None, alpha=0.0, beta=1.0,
                       norm_type=cv.NORM_MINMAX, dtype=cv.CV_32FC1)

    # compute gradient using sobel and magtitude
    grad_x = cv.Sobel(img, cv.CV_32FC1, 1, 0,
                      ksize=sobel_size)
    grad_y = cv.Sobel(img, cv.CV_32FC1, 0, 1,
                      ksize=sobel_size)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # normalize gradient
    mask = mag != 0
    grad_x[mask] /= mag[mask]
    grad_y[mask] /= mag[mask]

    # normalize magnitude
    mag = cv.normalize(mag, dst=None, alpha=0.0, beta=1.0,
                       norm_type=cv.NORM_MINMAX)

    # rotate gradient and get tangent vector
    flow_x, flow_y = -grad_y, grad_x

    # expand dimension in axis=0 for vectorizing
    flow = np.concatenate((flow_x[np.newaxis, ...],
                           flow_y[np.newaxis, ...]), axis=0)
    mag = mag[np.newaxis, ...]

    return flow, mag


def refine_flow(flow, mag, ksize):
    """ Refine edge tangent flow based on paper's
    equation. 根据论文方程细化边缘切线流。"""
    _, h, w = flow.shape

    # do padding
    p = ksize // 2
    flow = np.pad(flow, ((0, 0), (p, p), (p, p)))
    mag = np.pad(mag, ((0, 0), (p, p), (p, p)))

    # neighbors of each tangent vector in each window
    flow_neighbors = find_neighbors(flow, ksize, s=1, out_h=h, out_w=w)

    # centural tangent vector in each window
    flow_me = flow_neighbors[..., ksize // 2, ksize // 2]
    flow_me = flow_me[..., np.newaxis, np.newaxis]

    # compute dot
    dots = np.sum(flow_neighbors * flow_me, axis=2, keepdims=True)

    # compute phi
    phi = np.where(dots > 0, 1, -1)

    # compute wd, weight of direction
    wd = np.abs(dots)

    # compute wm, weight of magnitude
    mag_neighbors = find_neighbors(mag, ksize,
            s=1, out_h=h, out_w=w)
    mag_me = mag_neighbors[..., ksize // 2, ksize // 2]
    mag_me = mag_me[..., np.newaxis, np.newaxis]
    wm = (1 + np.tanh(mag_neighbors - mag_me)) / 2

    # compute ws, spatial weight
    ws = np.ones_like(wm)
    x, y = np.meshgrid(np.arange(ksize), np.arange(ksize))
    cx, cy = ksize // 2, ksize // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)[np.newaxis, ...]
    ws[:, :, dist >= ksize // 2] = 0

    # update flow
    flow = np.sum(phi * flow_neighbors * ws * wm * wd, axis=(3, 4))
    flow = np.transpose(flow, axes=(2, 0, 1))

    # normalize flow
    norm = np.sqrt(np.sum(flow ** 2, axis=0))
    mask = norm != 0
    flow[:, mask] /= norm[mask]

    return flow


def guass(x, sigma):
    """
    Returns guass filter's weight with respect to x and sigma.
    返回guass滤波器相对于x和sigma的权重。
    """
    return np.exp(-(x ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def make_gauss_filter(sigma, threshold=0.001):
    """ Returns a symetric gauss 1-d filter, max weight
    of output filter is less than threshold."""
    i = 0
    while guass(i, sigma) >= threshold:
        i = i + 1

    return guass(np.arange(-i, i + 1),
        sigma).astype('float32')


def shrink_array(a, center, width):
    """ Shrink an 1-D array with respect to center
    and width. """
    return a[-width + center: width + 1 + center]




