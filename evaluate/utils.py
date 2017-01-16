import numpy as np


def segment_iou(target_segment, candidate_segments):
    """
    Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.

    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def derive_recall_rate(ground_truth_segments, candidate_segments, tIoU_thresh):
    """
    Compute the recall rate of the target video
    Parameters
    ----------
    ground_truth_segments : 2d array
        Temporal target segment containing M x [starting, ending] times.

    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    recall_rate
    """
    ground_truth_num = ground_truth_segments.shape[0]
    recalled_count = 0.0
    for ground_truth_segment in ground_truth_segments:
        tIoU = segment_iou(ground_truth_segment, candidate_segments)
        sorted_tIoU = np.sort(tIoU)
        highest_detection_score = sorted_tIoU[-1]
        if highest_detection_score >= tIoU_thresh:
            recalled_count += 1

    recall_rate = recalled_count / ground_truth_num
    return recall_rate, recalled_count, ground_truth_num



def derive_overall_recall_rate(tIoU_thresh, results, print_info=False):
    if print_info: print('----------------------------------------------')
    total_recalled_count = 0.0
    total_ground_truth_num = 0.0
    for key in results.keys():
        ground_truth_segments = np.array(results[key]['ground_truth'])
        candidate_segments = np.array(results[key]['predict'])
        recall_rate, recalled_count, ground_truth_num = derive_recall_rate(ground_truth_segments, candidate_segments, tIoU_thresh)

        if print_info:
            print(key + '\t- local recall rate: ' + str(recall_rate) + '\t(' + str(recalled_count) + '/' + str(ground_truth_num) + ')')

        total_recalled_count += recalled_count
        total_ground_truth_num += ground_truth_num

    total_recall_rate = total_recalled_count / total_ground_truth_num
    if print_info:
        print
        print('Total recall rate: ' + str(total_recall_rate) + '\t(' + str(total_recalled_count) + '/' + str(total_ground_truth_num) + ')')
        print('tIoU_thresh: ' + str(tIoU_thresh))

    return total_recall_rate
