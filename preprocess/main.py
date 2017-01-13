import MSRII
from c3d import C3DFeatureNet


def main():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    c3dnet = C3DFeatureNet(feature_file='./MSRII-c3d-features.h5')

    c3dnet.start(dataset)


if __name__ == '__main__':
    main()
