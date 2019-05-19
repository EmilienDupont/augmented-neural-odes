import glob
import imageio
import numpy as np
import torch
from math import pi
from random import random
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
from torchvision import datasets, transforms


class Data1D(Dataset):
    """1D dimensional data used to demonstrate there are functions ODE flows
    cannot represent. Corresponds to g_1d(x) in the paper if target_flip is
    True.

    Parameters
    ----------
    num_points : int
        Number of points in dataset.

    target_flip : bool
        If True, sign of target is flipped.

    noise_scale : float
        Defaults to 0.0 (i.e. no noise). Otherwise, corresponds to standard
        deviation of white noise added to each point.
    """
    def __init__(self, num_points, target_flip=False, noise_scale=0.0):
        self.num_points = num_points
        self.target_flip = target_flip
        self.noise_scale = noise_scale
        self.data = []
        self.targets = []

        noise = Normal(loc=0., scale=self.noise_scale)

        for _ in range(num_points):
            if random() > 0.5:
                data_point = 1.0
                target = 1.0
            else:
                data_point = -1.0
                target = -1.0

            if self.target_flip:
                target *= -1

            if self.noise_scale > 0.0:
                data_point += noise.sample()

            self.data.append(torch.Tensor([data_point]))
            self.targets.append(torch.Tensor([target]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.num_points


class ConcentricSphere(Dataset):
    """Dataset of concentric d-dimensional spheres. Points in the inner sphere
    are mapped to -1, while points in the outer sphere are mapped 1.

    Parameters
    ----------
    dim : int
        Dimension of spheres.

    inner_range : (float, float)
        Minimum and maximum radius of inner sphere. For example if inner_range
        is (1., 2.) then all points in inner sphere will lie a distance of
        between 1.0 and 2.0 from the origin.

    outer_range : (float, float)
        Minimum and maximum radius of outer sphere.

    num_points_inner : int
        Number of points in inner cluster

    num_points_outer : int
        Number of points in outer cluster
    """
    def __init__(self, dim, inner_range, outer_range, num_points_inner,
                 num_points_outer):
        self.dim = dim
        self.inner_range = inner_range
        self.outer_range = outer_range
        self.num_points_inner = num_points_inner
        self.num_points_outer = num_points_outer

        self.data = []
        self.targets = []

        # Generate data for inner sphere
        for _ in range(self.num_points_inner):
            self.data.append(
                random_point_in_sphere(dim, inner_range[0], inner_range[1])
            )
            self.targets.append(torch.Tensor([-1]))

        # Generate data for outer sphere
        for _ in range(self.num_points_outer):
            self.data.append(
                random_point_in_sphere(dim, outer_range[0], outer_range[1])
            )
            self.targets.append(torch.Tensor([1]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class ShiftedSines(Dataset):
    """Dataset of two shifted sine curves. Points from the curve shifted upward
    are mapped to 1, while points from the curve shifted downward are mapped to
    1.

    Parameters
    ----------
    dim : int
        Dimension of datapoints.

    shift : float
        Size of shift/gap between the two curves.

    num_points_upper : int
        Number of points in upper curve.

    num_points_lower : int
        Number of points in lower curve.

    noise_scale : float
        Defaults to 0.0 (i.e. no noise). Otherwise, corresponds to standard
        deviation of white noise added to each point.
    """
    def __init__(self, dim, shift, num_points_upper, num_points_lower,
                 noise_scale):
        self.dim = dim
        self.shift = shift
        self.num_points_upper = num_points_upper
        self.num_points_lower = num_points_lower
        self.noise_scale = noise_scale

        noise = Normal(loc=0., scale=self.noise_scale)

        self.data = []
        self.targets = []

        # Generate data for upper curve and lower curve
        for i in range(self.num_points_upper + self.num_points_lower):
            if i < self.num_points_upper:
                label = 1
                y_shift = shift / 2.
            else:
                label = -1
                y_shift = - shift / 2.

            x = 2 * torch.rand(1) - 1  # Random point between -1 and 1
            y = torch.sin(pi * x) + noise.sample() + y_shift

            if self.dim == 1:
                self.data.append(torch.Tensor([y]))
            elif self.dim == 2:
                self.data.append(torch.cat([x, y]))
            else:
                random_higher_dims = 2 * torch.rand(self.dim - 2) - 1
                self.data.append(torch.cat([x, y, random_higher_dims]))

            self.targets.append(torch.Tensor([label]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


def random_point_in_sphere(dim, min_radius, max_radius):
    """Returns a point sampled uniformly at random from a sphere if min_radius
    is 0. Else samples a point approximately uniformly on annulus.

    Parameters
    ----------
    dim : int
        Dimension of sphere

    min_radius : float
        Minimum distance of sampled point from origin.

    max_radius : float
        Maximum distance of sampled point from origin.
    """
    # Sample distance of point from origin
    unif = random()
    distance = (max_radius - min_radius) * (unif ** (1. / dim)) + min_radius
    # Sample direction of point away from origin
    direction = torch.randn(dim)
    unit_direction = direction / torch.norm(direction, 2)
    return distance * unit_direction


def dataset_to_numpy(dataset):
    """Converts a Pytorch Dataset to the typical X, y numpy arrays expected by
    scikit-learn. This is useful for performing hyperparameter search.

    dataset : torch.utils.data.Dataset
        One of ConcentricSphere and ShiftedSines
    """
    num_points = len(dataset)
    X = np.zeros((num_points, dataset.dim))
    y = np.zeros((num_points, 1))
    for i in range(num_points):
        X[i] = dataset.data[i].numpy()
        y[i] = dataset.targets[i].item()
    return X.astype('float32'), y.astype('float32')


def mnist(batch_size=64, size=28, path_to_data='../../mnist_data'):
    """MNIST dataloader with (28, 28) images.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def cifar10(batch_size=64, size=32, path_to_data='../../cifar10_data'):
    """CIFAR10 dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 32 for no resizing.

    path_to_data : string
        Path to CIFAR10 data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.CIFAR10(path_to_data, train=True, download=True,
                                  transform=all_transforms)
    test_data = datasets.CIFAR10(path_to_data, train=False,
                                 transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def tiny_imagenet(batch_size=64, path_to_data='../../tiny-imagenet-200/'):
    """Tiny ImageNet dataloader.

    Parameters
    ----------
    batch_size : int

    path_to_data : string
        Path to Tiny ImageNet data files root folder.
    """
    imagenet_data = TinyImageNet(root_folder=path_to_data,
                                 transform=transforms.ToTensor())
    imagenet_loader = DataLoader(imagenet_data, batch_size=batch_size,
                                 shuffle=True)
    return imagenet_loader


class TinyImageNet(Dataset):
    """Tiny ImageNet dataset (https://tiny-imagenet.herokuapp.com/), containing
    64 x 64 ImageNet images from 200 classes.

    Parameters
    ----------
    root_folder : string
        Root folder of Tiny ImageNet dataset.

    transform : torchvision.transforms
    """
    def __init__(self, root_folder='../../tiny-imagenet-200/', transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.imgs_and_classes = []  # Paths to images and their classes

        train_folder = root_folder + 'train/'
        class_folders = glob.glob(train_folder + '*')  # One folder for each image class

        for i, class_folder in enumerate(class_folders):
            image_paths = glob.glob(class_folder + '/images/*.JPEG')
            for image_path in image_paths:
                self.imgs_and_classes.append((image_path, i))

        self.transform = transform

    def __len__(self):
        return len(self.imgs_and_classes)

    def __getitem__(self, idx):
        img_path, label = self.imgs_and_classes[idx]
        img = imageio.imread(img_path)

        if self.transform:
            img = self.transform(img)

        # Some images are grayscale, convert to RGB
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img, label
