import numpy as np
import time
import argparse
import random
from tqdm import tqdm
from keras.datasets import mnist, cifar10, fashion_mnist, cifar100
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from scipy.io import loadmat

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", "fashion_mnist", "cifar100", "SVHN"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa, "Select either 'lsa' or 'dsa'"
    print(args)
    
    if args.d == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        # print(x_test.shape)
        # Load pre-trained model.
        model=load_model("/content/drive/MyDrive/sadl11/model/model_fashion_mnist_LeNet4.h5")
        # # # You can select some layers you want to test.
        # # You can select some layers you want to test.
        layer_names = list(np.load("/content/drive/MyDrive/sadl11/layer_names.npy"))
        # print("LLLLLLLLLLLLLLLLLLLLLLLL", layer_names)
    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        # Load pre-trained model.
       # # You can select some layers you want to test.
        # LeNet1
        # layer_names = ["conv2d_1"]
        #LeNet5
        # layer_names = ["activation_13"]
        model= load_model("/content/drive/MyDrive/sadl11/model/model_Cov.h5")
        # # You can select some layers you want to test.
        layer_names = list(np.load("/content/drive/MyDrive/sadl11/layer_names.npy"))
        # print("LLLLLLLLLLLLLLLLLLLLLLLL", layer_names)


        # # Load target set.
        # # x_target = np.load("./adv/adv_mnist_{}.npy".format(args.target))
    if args.d == "SVHN":
        train_raw = loadmat('/content/drive/MyDrive/Data/train_32x32.mat')
        test_raw = loadmat('/content/drive/MyDrive/Data/test_32x32.mat')
        x_train = np.array(train_raw['X'])
        x_test = np.array(test_raw['X'])
        y_train = train_raw['y']
        y_test = test_raw['y']
        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)
        x_test= x_test.reshape (-1,32,32,3)
        x_train= x_train.reshape (-1,32,32,3)
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_test = lb.fit_transform(y_test)
        
        model= load_model("/content/drive/MyDrive/sadl11/model/model_SVHN_LeNet5.h5")
        # # You can select some layers you want to test.
        layer_names = list(np.load("/content/drive/MyDrive/sadl11/layer_names.npy"))
        # print("LLLLLLLLLLLLLLLLLLLLLLLL", layer_names)
    
    elif args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # model = load_model("/content/drive/MyDrive/sadl11/model/model_cifar.h5")
        x_train = x_train.reshape(-1, 32, 32, 3)
        x_test = x_test.reshape(-1, 32, 32, 3)
        
        model= load_model("/content/drive/MyDrive/sadl11/model/model_Cov.h5")
        # # You can select some layers you want to test.
        layer_names = list(np.load("/content/drive/MyDrive/sadl11/layer_names.npy"))
        # print("LLLLLLL/LLLLLLLLLLLLLLLLL", layer_names)
        # layer_names = ["activation_3"]
        # cifar100
        # layer_names=["activation_18"]

    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)

    if args.lsa:
      x_testLSC=np.load("/content/drive/MyDrive/sadl11/tmp/x_tcovlsc.npy")
      # print("shaaaaaaaaaaaaaaa", x_testLSC.shape)
      test_lsa = fetch_lsa(model, x_train, x_testLSC, "test", layer_names, args)
      # if args.d == "fashion_mnist" or args.d == "mnist":
      test_cov1 = get_sc( np.amin(test_lsa), args.upper_bound, args.n_bucket, test_lsa) 

      # if args.d == "cifar" or args.d == "cifar100" or args.d == "SVHN": 
      #   test_cov1 = get_sc(-140, args.upper_bound, args.n_bucket, test_lsa) 
      np.save("/content/drive/MyDrive/sadl11/tmp/test_cov.npy",test_cov1)
      print("args.upper_bound, args.n_bucket", args.upper_bound, args.n_bucket)
      print(infog("{} LSC coverage: ".format("test") + str(test_cov1)))
        
# print(infog("ROC-AUC: " + str(auc * 100)))
    if args.dsa:
        x_testDSC=np.load("/content/drive/MyDrive/sadl11/tmp/x_tcovdsc.npy")
        test_dsa = fetch_dsa(model, x_train, x_testDSC, "test", layer_names, args)
        test_cov = get_sc(0, args.upper_bound, args.n_bucket, test_dsa) 
        np.save("/content/drive/MyDrive/sadl11/tmp/DSC_cov.npy",test_cov)
        print(infog("{} DSC coverage: ".format("test") + str(test_cov)))
       