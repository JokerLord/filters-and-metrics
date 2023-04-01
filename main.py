import argparse
import numpy as np
from skimage.io import imread, imsave
from numpy.typing import NDArray
from scipy.signal import gaussian
from scipy.ndimage import convolve

L = 255


def read_image(filename: str) -> NDArray:
    image = imread(filename).astype(np.float64)
    return image


def mse(img1: NDArray, img2: NDArray) -> float:
    if img1.ndim == 3:
        img1 = img1[:, :, 0]
    if img2.ndim == 3:
        img2 = img2[:, :, 0]
    return np.mean((img1 - img2) ** 2) * 1
 # type: ignore # type: ignore # type: ignore # type: ignore # type: ignore

def psnr(img1: NDArray, img2: NDArray) -> float:
    if img1.ndim == 3:
        img1 = img1[:, :, 0]
    if img2.ndim == 3:
        img2 = img2[:, :, 0]
    mse_val = mse(img1, img2)
    if mse_val == 0:
        raise ValueError
    return 10 * np.log10(L ** 2 / mse_val)


def ssim(img1: NDArray, img2: NDArray, k1: float = 0.01,
         k2: float = 0.03) -> float:
    if img1.ndim == 3:
        img1 = img1[:, :, 0]
    if img2.ndim == 3:
        img2 = img2[:, :, 0]

    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    var1 = np.var(img1)
    var2 = np.var(img2)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    cov = np.cov(img1.ravel(), img2.ravel())[0, 1]
    return (2 * mean1 * mean2 + c1) * (2 * cov + c2) / \
           ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))


def save_image(filename: str, image: NDArray) -> None:
    imsave(filename, np.clip(image, 0, 255).astype(np.uint8))


def median_filter(image: NDArray, rad: int) -> NDArray:
    # padding
    padding = ((rad, rad), (rad, rad))
    if image.ndim == 3:
        padding += ((0, 0),)
    image = np.pad(image, padding, 'edge')

    filtered_img = []
    for i in range(rad, image.shape[0] - rad):
        line = []
        for j in range(rad, image.shape[1] - rad):
            window = image[i - rad: i + rad + 1, j - rad: j + rad + 1]
            val = np.median(window, axis=(0, 1))
            line.append(val)
        filtered_img.append(line)
    return np.array(filtered_img)


def gauss_kernel(size: int, sigma: float) -> NDArray:
    gauss_kernel1d = gaussian(size, sigma)
    gauss_kernel2d = np.outer(gauss_kernel1d, gauss_kernel1d)
    return gauss_kernel2d / np.sum(gauss_kernel2d)


def gauss_filter(image: NDArray, sigma: float) -> NDArray:
    is_colored = False
    if image.ndim == 3:
        is_colored = True
        image = image[:, :, 0]

    rad = int(3 * sigma)
    gauss_kernel2d = gauss_kernel(2 * rad + 1, sigma)
    filtered_img = convolve(image, gauss_kernel2d, mode="nearest")

    if is_colored:
        filtered_img = np.stack([filtered_img, filtered_img, filtered_img],
                                axis=2)
    return filtered_img


def gauss_func(x: NDArray, center: float, sigma: float) -> NDArray:
    return np.exp(-(x - center) ** 2 / (2 * sigma ** 2))


def bilateral_filter(image: NDArray, sigma_d: float,
                     sigma_r: float) -> NDArray:
    is_colored = False
    if image.ndim == 3:
        is_colored = True
        image = image[:, :, 0]

    rad = int(3 * sigma_d)
    # padding
    padding = ((rad, rad), (rad, rad))
    image = np.pad(image, padding, 'edge')

    gauss_kernel2d = gauss_kernel(2 * rad + 1, sigma_d)

    filtered_img = []
    for i in range(rad, image.shape[0] - rad):
        line = []
        for j in range(rad, image.shape[1] - rad):
            window = image[i - rad: i + rad + 1, j - rad: j + rad + 1]
            weights = (gauss_func(window, window[rad, rad], sigma_r) *
                       gauss_kernel2d)
            val = np.sum(weights * window) / np.sum(weights)
            line.append(val)
        filtered_img.append(line)
    filtered_img = np.array(filtered_img)
    if is_colored:
        filtered_img = np.stack([filtered_img, filtered_img, filtered_img],
                                axis=2)
    return filtered_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_mse = subparsers.add_parser("mse", help="Calculates MSE metric")
    parser_mse.add_argument("input_file_1", type=str)
    parser_mse.add_argument("input_file_2", type=str)

    parser_psnr = subparsers.add_parser("psnr", help="Calculates PSNR metric")
    parser_psnr.add_argument("input_file_1", type=str)
    parser_psnr.add_argument("input_file_2", type=str)

    parser_ssim = subparsers.add_parser("ssim", help="Calculates SSIM metric")
    parser_ssim.add_argument("input_file_1", type=str)
    parser_ssim.add_argument("input_file_2", type=str)

    parser_median = subparsers.add_parser("median", help="Median filter")
    parser_median.add_argument("rad", type=int)
    parser_median.add_argument("input_file", type=str)
    parser_median.add_argument("output_file", type=str)

    parser_gauss = subparsers.add_parser("gauss", help="Gaussian filter")
    parser_gauss.add_argument("sigma_d", type=float)
    parser_gauss.add_argument("input_file", type=str)
    parser_gauss.add_argument("output_file", type=str)

    parser_bilateral = subparsers.add_parser("bilateral",
                                             help="Bilateral filter")
    parser_bilateral.add_argument("sigma_d", type=float)
    parser_bilateral.add_argument("sigma_r", type=float)
    parser_bilateral.add_argument("input_file", type=str)
    parser_bilateral.add_argument("output_file", type=str)

    args = parser.parse_args()

    if args.command == "mse":
        img1 = read_image(args.input_file_1)
        img2 = read_image(args.input_file_2)
        print(mse(img1, img2))
    elif args.command == "psnr":
        img1 = read_image(args.input_file_1)
        img2 = read_image(args.input_file_2)
        print(psnr(img1, img2))
    elif args.command == "ssim":
        img1 = read_image(args.input_file_1)
        img2 = read_image(args.input_file_2)
        print(ssim(img1, img2))
    elif args.command == "median":
        image = read_image(args.input_file)
        filtered_img = median_filter(image, args.rad)
        save_image(args.output_file, filtered_img)
    elif args.command == "gauss":
        image = read_image(args.input_file)
        filtered_img = gauss_filter(image, args.sigma_d)
        save_image(args.output_file, filtered_img)
    elif args.command == "bilateral":
        image = read_image(args.input_file)
        filtered_img = bilateral_filter(image, args.sigma_d, args.sigma_r)
        save_image(args.output_file, filtered_img)
