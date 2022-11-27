from imutils.video import VideoStream
import cv2
import face_recognition
import time
import imutils
import pickle


def video_recognition(encodings_path, display, detection_method):

    names = []

    # load encodings
    print("Loading encodings...")
    data = pickle.loads(open(encodings_path, "rb").read())

    # start video stream
    print("Starting camera stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        face_boxes = face_recognition.face_locations(rgb,
                                                     model=detection_method)
        encodings = face_recognition.face_encodings(rgb, face_boxes)

        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "not found"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(face_boxes, names):
            top = int(top * r)
            bottom = int(bottom * r)
            left = int(left * r)
            right = int(right * r)
            cv2.rectangle(frame, (right, top), (left, bottom),
                          (0, 153, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 153, 0), 2)

        if display > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("w"):
                break

        # reset names every frame
        names = []

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    video_recognition("/Users/faizanrasool/School/ObjectDetection/face_recognition/recognition/encodings.pickle",
                      1, "hog")
