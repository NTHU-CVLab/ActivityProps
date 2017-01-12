import MSRII


def main():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    for _ in range(MSRII.Dataset.NUM_VIDEOS):
        dataset.write_video(dataset.take())

if __name__ == '__main__':
    main()
