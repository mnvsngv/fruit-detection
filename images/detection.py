import os

import cv2
import numpy as np


def object_detection(img, rects, model, fruits_db, size):
    # Collect the labels in text format
    id_to_labels = {i: v for i, v in
                    enumerate(os.listdir(fruits_db.train_dir))}

    # Perform object detection on the
    images = [cv2.resize(img[y:y + h, x:x + w], (size[0], size[1]))
              for x, y, w, h in rects[0:20]]
    results = model.predict(np.array(images))

    # Select predictions which have confidence of more than 95% and display
    objects, types = np.where(results > 0.95)
    for i, object in enumerate(objects):
        x, y, w, h = rects[object]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
        cv2.putText(img, id_to_labels[types[i]], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0))
        print(i)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    return results
