import numpy as np
import torch
import matplotlib.pyplot as plt
from math import log, exp
from scipy.integrate import quad
import pickle
from mpl_toolkits.mplot3d import Axes3D


def show_images(images, cols=1, titles=None, vmin=0, vmax=1):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, vmin=vmin, vmax=vmax)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def visualize_1d_potential(p, d, spacing=2):
    xs = np.arange(d.values[0], d.values[1], spacing)
    xs = xs.reshape(-1, 1)

    ys = p.batch_call(xs)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys)
    ax.set_xlabel('x')
    ax.set_ylabel('value')
    plt.show()


def visualize_2d_potential(p, d1, d2, spacing=2):
    d1 = np.arange(d1.values[0], d1.values[1], spacing)
    d2 = np.arange(d2.values[0], d2.values[1], spacing)

    x1, x2 = np.meshgrid(d1, d2)
    xs = np.array([x1, x2]).T.reshape(-1, 2)

    ys = p.batch_call(xs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, ys)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('value')
    plt.show()


def visualize_1d_potential_torch(p, d, spacing=2):
    device = next(p.nn.parameters()).device

    xs = torch.arange(d.values[0], d.values[1], spacing, device=device)
    xs = xs.reshape(-1, 1)

    with torch.no_grad():
        ys = p.batch_call(xs)

    xs = xs.cpu().numpy()
    ys = ys.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys)
    ax.set_xlabel('x')
    ax.set_ylabel('value')
    plt.show()


def visualize_2d_potential_torch(p, d1, d2, spacing=2):
    device = next(p.nn.parameters()).device

    d1 = np.arange(d1.values[0], d1.values[1], spacing)
    d2 = np.arange(d2.values[0], d2.values[1], spacing)

    x1, x2 = np.meshgrid(d1, d2)
    xs = np.array([x1, x2]).T.reshape(-1, 2)

    xs = torch.FloatTensor(xs).to(device)

    with torch.no_grad():
        ys = p.batch_call(xs)

    ys = ys.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, ys)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('value')
    plt.show()


def save(f, *objects):
    with open(f, 'wb') as file:
        pickle.dump(objects, file)


def load(f):
    with open(f, 'rb') as file:
        return pickle.load(file)
