import numpy as np
import cv2 as cv
import time
import os
from functions import *
from utils import *


def run(img, sobel_size=5, etf_iter=4, etf_size=7,
    fdog_iter=3, thresh=0.001, sigma_c=1.0, rho=0.997,
    sigma_m=3.0, tau=0.907):
    """
    Running coherent line drawing on input image.

    Parameters:
    ----------
    - img : ndarray
        Input image, with shape = (h, w, c).
    
    - sobel_size : int, default = 5
        Size of sobel filter, sobel filter will be used to compute
        gradient.
    
    - etf_iter : int, default = 4
        Iteration times of refining edge tangent flow.
    
    - etf_size : int, default = 7
        Size of etf filter.
    
    - fdog_iter : int, default = 3
        Iteration times of applying fdog on input image.
    
    - thresh : float, default = 0.001
        Threshold of guass filter's value, this is not an important
        parameter.
    
    - sigma_c : float, default = 1.0
        Standard variance of one gaussian filter of dog filter,
        another's standard variance will be set to 1.6 * sigma_c.
    
    - rho : float, default = 0.997
        Dog = gauss_c - rho * gauss_s.
    
    - sigma_m : float, default = 3.0
        Standard variance of gaussian filter.
    
    - tau : float, default=0.907
        Threshold of edge map.

    Returns:
    -------
    - edge : ndarray
        Edge map of input image, data type is float32 and pixel's
        range is clipped to [0, 255].
    """
    # convert to gray image
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # initialize edge tangent flow
    flow, mag = initialze_flow(img, sobel_size)
    print("flow shape - 1:", flow.shape)
    print("mag shape:", mag.shape)

    # refine edge tangent flow
    for i in range(etf_iter):
        start = time.perf_counter()
        flow = refine_flow(flow, mag, ksize=etf_size)
        end = time.perf_counter()
        print(f"smoothing edge tangent flow, iteration {i + 1}, "
                f"time cost = {end - start:<6f}s")

    print("flow shape -2 " + (str)(flow.shape))

    # do fdog
    for i in range(fdog_iter):
        start = time.perf_counter()
        edge = detect_edge(img, flow, thresh=thresh,
            sigma_c=sigma_c, rho=rho, sigma_m=sigma_m, tau=tau)
        img[edge == 0] = 0
        img = cv.GaussianBlur(img, ksize=(3, 3), sigmaX=0, sigmaY=0)
        end = time.perf_counter()
        print(f"applying fdog, iteration {i + 1}, "
                f"time cost = {end - start:<6f}s")

    return detect_edge(img, flow, thresh=thresh,
        sigma_c=sigma_c, rho=rho, sigma_m=sigma_m, tau=tau),flow


if __name__ == "__main__":
    tests = [
        '1.jpg', '2.jpg',
        '3.jpg', '4.jpg',
        '5.jpg', '6.jpg',
    ]

    resize_value = 1.0 # Resize The Image

    for test in tests:
        print(f"running on test {test}")
        # read image
        img = cv.imread(os.path.join('benchmarks', test))

        # shrink image if its size is considerable (500?)
        shape = img.shape[:-1][::-1]

        if any(map(lambda sz: sz > 500, shape)):
            img = cv.resize(img,
                tuple(map(lambda sz: int(sz * resize_value), shape)))  # the resize module of the input image
        print(f"input shape = {shape}")

        # run on this image and return edge map
        edge,flow = run(
            img=img, sobel_size=5,
            etf_iter=4, etf_size=7,
            fdog_iter=2, sigma_c=0.3, rho=0.997, sigma_m=3.0,
            tau=0.997
        )
        print("{}-{}".format('- ' * 10, ' -' * 9))

        # save the flow
        np.save(os.path.join('benchmarks', test.split('.')[0] + '_flow.npy'), flow)

        # convert edge to 3-channel image
        edge = cv.cvtColor(edge.astype(np.uint8), cv.COLOR_GRAY2RGB)

        #cat the img and the edge
        img = np.concatenate((img,edge),axis=1)
        flow = np.concatenate((flow[0,:,:],flow[1,:,:]),axis=1)

        # cv_imshow(img, title="output", wait=0.1)
        # cv_imshow(flow, title="flow", wait=0)

        # show origin image and edge map
        # cv_imshow(img, title="input", wait=0.1)
        # cv_imshow(flow, title="flow", wait=0.1)
        # cv_imshow(edge.astype('uint8'), title="output", wait=0,
        #     move=(int(img.shape[1] * 1.5), 0))

        # save result
        cv.imwrite(f"edge_{test}", img)
        flow = flow * 255
        cv.imwrite(f"flow_{test}", flow)
        cv.destroyAllWindows()