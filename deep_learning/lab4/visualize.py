import numpy as np
import torch
from dataset import MNISTMetricDataset
from model import SimpleMetricEmbedding
from matplotlib import pyplot as plt

def get_colormap():
    # Cityscapes colormap for first 10 classes
    colormap = np.zeros((10, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    return colormap

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")
    emb_size = 32

    first_model = SimpleMetricEmbedding(1, emb_size).to(device)
    first_model.load_state_dict(torch.load('first_model.pth'))

    model_remove0 = SimpleMetricEmbedding(1, emb_size).to(device)
    model_remove0.load_state_dict(torch.load('remove0_model.pth'))

    colormap = get_colormap()
    mnist_download_root = "./mnist/"
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    X = ds_test.images
    Y = ds_test.targets

    print("Fitting PCA directly from images...")
    test_img_rep2d = torch.pca_lowrank(ds_test.images.view(-1, 28 * 28), 2)[0]
    plt.scatter(test_img_rep2d[:, 0], test_img_rep2d[:, 1], c=colormap[Y] / 255., s=5)
    plt.title("PCA of Test Images")
    plt.show()
    plt.figure()

    print("Fitting PCA from feature representation -> all digits")
    with torch.no_grad():
        first_model.eval()
        test_rep_all_digits = first_model.get_features(X.unsqueeze(1).to(device))
        test_rep2d_all_digits = torch.pca_lowrank(test_rep_all_digits, 2)[0].cpu()
        plt.scatter(test_rep2d_all_digits[:, 0], test_rep2d_all_digits[:, 1], c=colormap[Y] / 255., s=5)
        plt.title("PCA of Feature Representation -> all digits")
        plt.show()
    plt.figure()

    print("Fitting PCA from feature representation -> digit 0 removed")
    with torch.no_grad():
        model_remove0.eval()
        test_rep_no_zero = model_remove0.get_features(X.unsqueeze(1).to(device))
        test_rep2d_no_zero = torch.pca_lowrank(test_rep_no_zero, 2)[0].cpu()
        plt.scatter(test_rep2d_no_zero[:, 0], test_rep2d_no_zero[:, 1], c=colormap[Y] / 255., s=5)
        plt.title("PCA of Feature Representation -> digit 0 removed")
        plt.show()
