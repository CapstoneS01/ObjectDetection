import face_recognition
import cv2
import pickle
import os
import sys


def encode_faces(dataset_path, encoding_path, detection_method):

    defined_encodings = []
    defined_names = []
    # store paths of images
    image_paths = []
    for path in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, path)):
            image_paths.append(path)

    if ".DS_Store" in image_paths:
        image_paths.remove(".DS_Store")

    for (i, imagePath) in enumerate(image_paths):
        # get person name from image path
        name = imagePath.split(os.path.sep)[0].split("_")[0]
        image = cv2.imread("./dataset/" + imagePath.split(os.path.sep)[0])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect face in image and create a box around it
        face_boxes = face_recognition.face_locations(
            rgb, model=detection_method)
        # get facial encodings
        encodings = face_recognition.face_encodings(rgb, face_boxes)
        for encoding in encodings:
            # add encodings and names of faces to list
            defined_encodings.append(encoding)
            defined_names.append(name)

    data = {"encodings": defined_encodings, "names": defined_names}
    file = open(encoding_path, "wb")
    file.write(pickle.dumps(data))
    file.close()


if __name__ == "__main__":
    # print("Beginning encoding...")
    encode_faces("../../Web/server/dataset",
                 "../../ObjectDetection/face_recognition/encodings.pickle", "cnn")
    print("Encoding complete")
    # sys.stdout.flush()
