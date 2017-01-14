from preprocess import MSRII
from preprocess.c3d import C3DFeatureNet
from preprocess.video import Video
from network.model import FC4Net


def main():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    c3dnet = C3DFeatureNet(feature_file='preprocess/MSRII-c3d-features.h5')

    c3dnet.load()
    c3dnet.start(dataset)


def test():
    video = Video('/data-disk/MSRII/videos/1.avi')

    c3dnet = C3DFeatureNet(feature_file=None)
    c3dnet.load()
    y = c3dnet.extract_feature(video)

    fc4net = FC4Net()
    fc4net.load_weights('network/FC4Net_weights.h5')
    pred = fc4net.predict(y, batch_size=32)

    print(pred)

if __name__ == '__main__':
    test()
