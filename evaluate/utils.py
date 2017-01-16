import numpy as np


def non_maximum_suppression(pred, tol_frames=16, prob_ths=.5):
    CLIP_FRAMES = 16
    p = pred.reshape(len(pred))
    candidate = np.where(p > prob_ths)

    start = np.multiply(candidate, CLIP_FRAMES).tolist()[0]
    end = (np.multiply(candidate, CLIP_FRAMES) + CLIP_FRAMES).tolist()[0]

    predict = []
    skip = [0 for _ in range(len(start))]
    for i, p in enumerate(zip(start, end)):
        s, e = p
        if skip[i]:
            continue
        next_idx = i + 1 if i + 1 < len(start) else -1
        for j, next_s in enumerate(start[next_idx:]):
            if e + tol_frames >= next_s:
                e = end[next_idx + j]
                skip[next_idx + j] = 1
        predict.append((s, e))

    return predict