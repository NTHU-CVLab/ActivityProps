import MSRII


def main():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    video1 = dataset.take()
    video2 = dataset.take()
    print(video1[0])
    print(video2[0])

if __name__ == '__main__':
    main()

