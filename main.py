{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled34.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOGYHDM/ohRCuI7ZKL0bt45",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/batraamul/Image_Noise_Identification/blob/main/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1Ifjr9z_07ur"
      },
      "source": [
        "import numpy as np\n",
        "import os\n",
        "import numpy as np\n",
        "from numpy.fft import fft2, ifft2\n",
        "from scipy.signal import gaussian, convolve2d\n",
        "import matplotlib.pyplot as plt\n"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WgSe-FYz8RUU",
        "outputId": "6d5c79f9-c23e-48fd-caf5-151aa33d463c"
      },
      "source": [
        "!git clone https://github.com/batraamul/Image_Noise_Identification.git"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Cloning into 'Image_Noise_Identification'...\n",
            "remote: Enumerating objects: 86, done.\u001b[K\n",
            "remote: Counting objects: 100% (86/86), done.\u001b[K\n",
            "remote: Compressing objects: 100% (70/70), done.\u001b[K\n",
            "remote: Total 86 (delta 4), reused 0 (delta 0), pack-reused 0\u001b[K\n",
            "Unpacking objects: 100% (86/86), done.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8ylz1zL91kxh"
      },
      "source": [
        "import cv2\n",
        "import os\n",
        "import skimage.filters\n",
        "from skimage.util import random_noise\n",
        "from skimage.io import imread\n",
        "from skimage import color, data, restoration\n"
      ],
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dPhPltMH2C6U"
      },
      "source": [
        "# Initializing the folders address under various scenarioes\n",
        "folder = \"/content/Image_Noise_Identification/Data/Images/HR_Images\"\n",
        "folder_ga=\"/content/Image_Noise_Identification/Data/Images/Noisy Images/Gaussian\"\n",
        "folder_sp=\"/content/Image_Noise_Identification/Data/Images/Noisy Images/SP\"\n",
        "folder_ga_sp=\"/content/Image_Noise_Identification/Data/Images/Noisy Images/Combined\"\n",
        "filtered_HR_W=\"/content/Image_Noise_Identification/Data/Images/Filtered_Images/HR_Filtered\"\n",
        "filtered_ga_W=\"/content/Image_Noise_Identification/Data/Images/Filtered_Images/Gaussian_Filtered\"\n",
        "fileter_sp_W=\"/content/Image_Noise_Identification/Data/Images/Filtered_Images/SP\"\n",
        "filtered_combined_W=\"/content/Image_Noise_Identification/Data/Images/Filtered_Images/Combined\"\n",
        "filtered_HR_M=\"/content/Image_Noise_Identification/Data/Images/Filtered_Images/HR_Filtered\"\n",
        "filtered_ga_M=\"/content/Image_Noise_Identification/Data/Images/Filtered_Images/Gaussian_Filtered\"\n",
        "fileter_sp_M=\"/content/Image_Noise_Identification/Data/Images/Filtered_Images/SP\"\n",
        "filtered_combined_M=\"/content/Image_Noise_Identification/Data/Images/Filtered_Images/Combined\"\n"
      ],
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aQcH-2jr8lJp"
      },
      "source": [
        "# Self compiled Function Definations \n",
        "\n",
        "def blur(img, kernel_size = 3):\n",
        "\tdummy = np.copy(img)\n",
        "\th = np.eye(kernel_size) / kernel_size\n",
        "\tdummy = convolve2d(dummy, h, mode = 'valid')\n",
        "\treturn dummy\n",
        "\n",
        "def add_gaussian_noise(img, sigma):\n",
        "\tgauss = np.random.normal(0, sigma, np.shape(img))\n",
        "\tnoisy_img = img + gauss\n",
        "\tnoisy_img[noisy_img < 0] = 0\n",
        "\tnoisy_img[noisy_img > 255] = 255\n",
        "\treturn noisy_img\n",
        "\n",
        "def wiener_filter(img, kernel, K):\n",
        "\tkernel /= np.sum(kernel)\n",
        "\tdummy = np.copy(img)\n",
        "\tdummy = fft2(dummy)\n",
        "\tkernel = fft2(kernel, s = img.shape)\n",
        "\tkernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)\n",
        "\tdummy = dummy * kernel\n",
        "\tdummy = np.abs(ifft2(dummy))\n",
        "\treturn dummy\n",
        "\n",
        "def gaussian_kernel(kernel_size = 3):\n",
        "\th = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)\n",
        "\th = np.dot(h, h.transpose())\n",
        "\th /= np.sum(h)\n",
        "\treturn h\n",
        "\n",
        "def rgb2gray(rgb):\n",
        "\treturn np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])"
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 350
        },
        "id": "9Jk9DRBi2XWV",
        "outputId": "5caefacb-8f6f-47fa-e074-493ef157efb1"
      },
      "source": [
        "# Picking up HR images one by one to process \n",
        "for filename in os.listdir(folder): \n",
        "     img = cv2.imread(os.path.join(folder, filename))\n",
        "\n",
        "     # 1. GENERATING NOISY IMAGES  \n",
        "     # 1.2. Adding Gaussian Noise \n",
        "     ga_noise=random_noise(img,mode='gaussian',seed=None,clip=True)\n",
        "     image_gaussian=np.array(255*ga_noise, dtype = 'uint8')\n",
        "     # 1.3. Adding Salt and Pepper Noise \n",
        "     sp_noise=random_noise(img, mode='s&p',amount=0.1)\n",
        "     image_sp=np.array(255*sp_noise, dtype = 'uint8')\n",
        "     # 1.4. Adding combined Noise \n",
        "     g_sp_noise=random_noise(image_gaussian, mode='s&p',amount=0.1)\n",
        "     image_g_sp=np.array(255*g_sp_noise, dtype = 'uint8')\n",
        "\n",
        "     # 2. APPLYING FILTERS TO THE IMAGES  \n",
        "     # 2.1. initialize the kernel \n",
        "     kernel = gaussian_kernel(3)\n",
        "     # 2.2. Generate Grey Scale Images\n",
        "     img_grey_scale_HR=rgb2gray(img)\n",
        "     img_grey_scale_Gaussian=rgb2gray(image_gaussian)\n",
        "     img_grey_scale_SandP=rgb2gray(image_sp)\n",
        "     img_grey_scale_Combined=rgb2gray(image_g_sp)\n",
        "     #2.3.  Apply Wiener Filter\n",
        "     # 2.3.1. HR Images\n",
        "     wiener_HR= wiener_filter(img_grey_scale_HR, kernel, K = 10)\n",
        "     # 2.3.2. images with Gaussian Noise \n",
        "     wiener_G= wiener_filter(img_grey_scale_Gaussian, kernel, K = 10)\n",
        "     # 2.3.3. Noisy images with SP Noise \n",
        "     wiener_SP= wiener_filter(img_grey_scale_SandP, kernel, K = 10)\n",
        "     # 2.3.4. Noisy images with Combined Noise \n",
        "     wiener_Combined= wiener_filter(img_grey_scale_Combined, kernel, K = 10)\n",
        "     # 2.4. Apply Median Filter\n",
        "     # 2.4.1. HR Images\n",
        "     median_HR= skimage.filters.median(img)\n",
        "     # 2.4.2. Noisy images with Gaussian Noise \n",
        "     median_G= skimage.filters.median(image_gaussian)\n",
        "     # 2.4.3. Noisy images with SP Noise \n",
        "     median_sp= skimage.filters.median(image_sp)\n",
        "     #2.4.4. Noisy images with Combined Noise\n",
        "     median_combined= skimage.filters.median(image_g_sp)\n",
        "\n",
        "     # 3. SAVING FILES \n",
        "     #3.1. Saving Images with Added Noise \n",
        "     cv2.imwrite(os.path.join(folder_ga, \"G_\" + filename), image_gaussian)\n",
        "     cv2.imwrite(os.path.join(folder_sp, \"SP_\" + filename), image_sp)\n",
        "     cv2.imwrite(os.path.join(folder_ga_sp,\"G_SP_\" + filename), image_g_sp)\n",
        "     #3.2. Saving Images with applied Wiener_Filter \n",
        "     cv2.imwrite(os.path.join(filtered_HR_W, \"W_HR_\" + filename), wiener_HR)\n",
        "     cv2.imwrite(os.path.join(filtered_ga_W, \"W_G_\" + filename), wiener_G)\n",
        "     cv2.imwrite(os.path.join(fileter_sp_W, \"W_SP_\" + filename), wiener_SP)\n",
        "     cv2.imwrite(os.path.join(filtered_combined_W,\"W_COMBINED_\" + filename), wiener_combined)\n",
        "     #3.3. Saving Images with applied Median_Filter \n",
        "     cv2.imwrite(os.path.join(filtered_HR_M, \"M_HR_\" + filename), median_HR)\n",
        "     cv2.imwrite(os.path.join(filtered_ga_M, \"M_G_\" + filename), median_G)\n",
        "     cv2.imwrite(os.path.join(fileter_sp_M, \"M_SP_\" + filename), median_sp)\n",
        "     cv2.imwrite(os.path.join(filtered_combined_M,\"M_COMBINED_\" + filename), median_combined)\n"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "error",
          "ename": "AttributeError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-14-d5ae262b17e4>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      5\u001b[0m      \u001b[0;31m# 1. GENERATING NOISY IMAGES\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m      \u001b[0;31m# 1.2. Adding Gaussian Noise\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 7\u001b[0;31m      \u001b[0mga_noise\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mrandom_noise\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimg\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'gaussian'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mclip\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      8\u001b[0m      \u001b[0mimage_gaussian\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m255\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mga_noise\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m'uint8'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m      \u001b[0;31m# 1.3. Adding Salt and Pepper Noise\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/skimage/util/noise.py\u001b[0m in \u001b[0;36mrandom_noise\u001b[0;34m(image, mode, seed, clip, **kwargs)\u001b[0m\n\u001b[1;32m     88\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     89\u001b[0m     \u001b[0;31m# Detect if a signed image was input\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 90\u001b[0;31m     \u001b[0;32mif\u001b[0m \u001b[0mimage\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m<\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     91\u001b[0m         \u001b[0mlow_clip\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m-\u001b[0m\u001b[0;36m1.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     92\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mAttributeError\u001b[0m: 'NoneType' object has no attribute 'min'"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "P0nx4dYP4Arr"
      },
      "source": [
        "# Calculating statistical components from the filtered images \n",
        "def image_parameters(Z):\n",
        "    \n",
        "    h,w = np.shape(Z)\n",
        "    \n",
        "# Z is the input image with added noise \n",
        "# here x gives row of the image matrix while y gives number of columns of the image matrix\n",
        "    x = range(w)\n",
        "    y = range(h)\n",
        "\n",
        "    X,Y = np.meshgrid(x,y)\n",
        "\n",
        "    #Centroid (mean)\n",
        "    cx = np.sum(Z*X)/np.sum(Z)\n",
        "    cy = np.sum(Z*Y)/np.sum(Z)\n",
        "\n",
        "    ###Standard deviation\n",
        "    x2 = (range(w) - cx)**2\n",
        "    y2 = (range(h) - cy)**2\n",
        "\n",
        "    X2,Y2 = np.meshgrid(x2,y2)\n",
        "\n",
        "    #Find the variance\n",
        "    vx = np.sum(Z*X2)/np.sum(Z)\n",
        "    vy = np.sum(Z*Y2)/np.sum(Z)\n",
        "\n",
        "    #SD is the sqrt of the variance\n",
        "    sx,sy = np.sqrt(vx),np.sqrt(vy)\n",
        "\n",
        "    ###Skewness\n",
        "    x3 = (range(w) - cx)**3\n",
        "    y3 = (range(h) - cy)**3\n",
        "\n",
        "    X3,Y3 = np.meshgrid(x3,y3)\n",
        "\n",
        "    #Find the thid central moment\n",
        "    m3x = np.sum(Z*X3)/np.sum(Z)\n",
        "    m3y = np.sum(Z*Y3)/np.sum(Z)\n",
        "\n",
        "    #Skewness is the third central moment divided by SD cubed\n",
        "    skx = m3x/sx**3\n",
        "    sky = m3y/sy**3\n",
        "\n",
        "    ###Kurtosis\n",
        "    x4 = (range(w) - cx)**4\n",
        "    y4 = (range(h) - cy)**4\n",
        "\n",
        "    X4,Y4 = np.meshgrid(x4,y4)\n",
        "\n",
        "    #Find the fourth central moment\n",
        "    m4x = np.sum(Z*X4)/np.sum(Z)\n",
        "    m4y = np.sum(Z*Y4)/np.sum(Z)\n",
        "\n",
        "    #Kurtosis is the fourth central moment divided by SD to the fourth power\n",
        "    kx = m4x/sx**4\n",
        "    ky = m4y/sy**4\n",
        "\n",
        "    return cx,cy,sx,sy,skx,sky,kx,ky"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GHMzPV6Ov7a8",
        "outputId": "4591fa3f-7cf9-4b0e-f1a0-9038933cd447"
      },
      "source": [
        "# Extracting image stats from HR Images \n",
        "text=os.path.join(folder_ga, \"g_\" + filename)\n",
        "print(text)\n",
        "hr=cv2.imread(os.path.join(folder_ga, \"g_\" + filename))\n",
        "hr=rgb2gray(hr)\n",
        "print(hr.shape)\n",
        "cx,cy,sx,sy,skx,sky,kx,ky= image_parameters(hr)\n",
        "print(cx,cy,sx,sy,skx,sky,kx,ky)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/content/folder_ga/g_Image_HR.JPG\n",
            "(3456, 4608)\n",
            "2295.8038541164924 1893.4400745458124 1300.923019626514 943.4370355893376 0.01018247670610294 -0.19880088897953174 1.88047663102744 1.9874786085340557\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2lLJHtD0wDV8"
      },
      "source": [
        "# Extracting image stats from Images having Gaussian Noise "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TUUuWmIIwuJg"
      },
      "source": [
        "# Extracting image stats from Images having Salt and Pepper Noise "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8oGWf3Rhwz07"
      },
      "source": [
        "# Extracting image stats from Images having Combination of Both Noises  "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7O4pSbcRw66F"
      },
      "source": [
        "# Creation of Data Frame and csv file to initiate Neural Network "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "up7jN3hvxCGM"
      },
      "source": [
        "# Creation of Neural Network "
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}