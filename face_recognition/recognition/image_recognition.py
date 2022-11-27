import pickle
import cv2
import face_recognition
import os
import argparse


def image_recognition(encodings_path, image_path, detection_method):
    names = []

    # load encodings
    print("Loading encodings...")
    data = pickle.loads(open(encodings_path, "rb").read())

    # load image and make it readable for cv2
    image_name = image_path.split(os.path.sep)[-1]
    image = cv2.imread("recognition/test/" + image_name)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Performing recognition...")
    face_boxes = face_recognition.face_locations(rgb,
                                                 model=detection_method)
    encodings = face_recognition.face_encodings(rgb, face_boxes)

    # match face in image to one in encodings
    for encoding in encodings:
        face_matches = face_recognition.compare_faces(data["encodings"],
                                                      encoding)
        name = "not found"

        # check dataset and get matches to input image
        if True in face_matches:
            matched_indexes = [i for (i, b) in enumerate(face_matches) if b]
            counts = {}
            for i in matched_indexes:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)

    # label all recognized faces
    for ((top, right, bottom, left), name) in zip(face_boxes, names):
        cv2.rectangle(image, (right, top), (left, bottom), (0, 153, 0), 3)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 153, 0), 2)

    cv2.imshow("img", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    image_recognition("/Users/faizanrasool/School/ObjectDetection/face_recognition/recognition/encodings.pickle",
                      "./test/test_image.jpg", "cnn")
