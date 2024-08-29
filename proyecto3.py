from neurons import neuron
import numpy as np
import struct, os, requests


def download_mnist():
    # URLs for the MNIST dataset
    base_url = 'https://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    # Download each file
    for key, filename in files.items():
        if not os.path.exists(filename):
            print(f'Downloading {filename}...')
            response = requests.get(base_url + filename)
            with open(filename, 'wb') as f:
                f.write(response.content)

def load_images(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of images
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(
            num_images, rows, cols)
        return images


def load_labels(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


if __name__ == '__main__':
    download_mnist()

    # Load the training data
    train_images = load_images('train-images-idx3-ubyte')
    train_labels = load_labels('train-labels-idx1-ubyte')

    # Load the test data
    test_images = load_images('t10k-images-idx3-ubyte')
    test_labels = load_labels('t10k-labels-idx1-ubyte')

    print(f'Train images shape: {train_images.shape}')
    print(f'Train labels shape: {train_labels.shape}')
    print(f'Test images shape: {test_images.shape}')
    print(f'Test labels shape: {test_labels.shape}')

    model = neuron.Neuron([])
