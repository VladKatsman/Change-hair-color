import os
from glob import glob


def main(root, num_splits):
    files_path = glob("{}/ijbc_videos_720p/*".format(root))
    dic = {}
    video_len = []
    for path in files_path:
        if not path.endswith('.pkl'):
            size = int(os.path.getsize(path))
            video_len.append(size)
            dic[size] = path
    video_len.sort()

    for i in range(num_splits):
        list_i = [dic[x] for x in video_len[i::num_splits]]
        path = os.path.join(root, '{}.txt'.format(i))
        print(len(list_i))
        with open(path, 'w') as t:
            for item in list_i:
                t.write("%s\n" % item)


if __name__ == '__main__':
    root = '/data/dev/datasets'
    num_splits = 8

    main(root, num_splits)
