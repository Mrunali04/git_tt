from sklearn.cluster import DBSCAN
import numpy as np
import sys
from .logging import Logger
import random


def cluster_images(encodings, img_paths):
    """

    :param encodings: list of encodings
    :param img_paths: list of image paths
    :return: dict containing encodings and images that are needed to be sent to server
    """
    try:
        clt = DBSCAN(metric="euclidean")
        clt.fit(encodings)
        # determine the total number of unique faces found in the dataset  current label ID
        label_ids = np.unique(clt.labels_)
        # loop over the unique face integers
        face_dict = {}
        for label_id in label_ids:
            # find all indexes into the `data` array that belong to the
            idxs = np.where(clt.labels_ == label_id)[0]
            if label_id == -1:
                face_dict[label_id] = {"encodings": [encodings[x] for x in idxs],
                                       "image_path": [img_paths[x] for x in idxs]}
            else:
                rand_ind = random.choice(idxs)
                face_dict[label_id] = {"encodings": [encodings[rand_ind]],
                                       "image_path": [img_paths[rand_ind]]}
            # loop over the sampled indexes
        return face_dict
    # for error logs

    except Exception as e:

        Logger.error_parse(e, sys.exc_info(), msg_level=3)
