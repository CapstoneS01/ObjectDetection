import face_recognition
import cv2
import pickle
import os


def encode_faces(dataset_path, encoding_path, detection_method):

    defined_encodings = []
    defined_names = []

    # store paths of images
    print("Quantifying faces...")
    image_paths = []
    for path in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, path)):
            image_paths.append(path)

    for (i, imagePath) in enumerate(image_paths):
        # get person name from image path
        print("Processing image {}/{}".format(i + 1,
                                              len(image_paths)))
        name = imagePath.split(os.path.sep)[0].split('_')[0]
        image = cv2.imread("recognition/dataset/" +
                           imagePath.split(os.path.sep)[0])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect face in image and create a box around it
        face_boxes = face_recognition.face_locations(rgb,
                                                     model=detection_method)
        # get facial encodings
        encodings = face_recognition.face_encodings(rgb, face_boxes)
        for encoding in encodings:
            # add encodings and names of faces to list
            defined_encodings.append(encoding)
            defined_names.append(name)

    print("Serializing encodings...")

    data = {"encodings": defined_encodings, "names": defined_names}
    f = open(encoding_path, "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == "__main__":
    encode_faces("/Users/faizanrasool/School/ObjectDetection/face_recognition/recognition/dataset",
                 "/Users/faizanrasool/School/ObjectDetection/face_recognition/recognition/encodings.pickle", "cnn")
