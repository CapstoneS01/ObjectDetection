import imutils.video
import pickle
import cv2
import face_recognition
from djitellopy import Tello
import face_recognition
import time


def skyhunt(encodings_path, detection_method):
    tello = Tello()
    tello.connect()
    tello.streamoff()
    tello.streamon()

    names = []
    data = pickle.loads(open(encodings_path, "rb").read())
    time.sleep(2.0)

    while (1):
        # read stream and convert to RGB
        frame_read = tello.get_frame_read()
        frame = frame_read.frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = imutils.resize(frame, width=750)

        # detect faces in frame and match with encodings
        video_ratio = frame.shape[1] / float(frame_rgb.shape[1])
        face_boxes = face_recognition.face_locations(frame_rgb,
                                                     model=detection_method)
        encodings = face_recognition.face_encodings(frame_rgb, face_boxes)

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

        # loop over the recognized faces and create a box around them
        for ((top, right, bottom, left), name) in zip(face_boxes, names):
            top = int(top * video_ratio)
            bottom = int(bottom * video_ratio)
            left = int(left * video_ratio)
            right = int(right * video_ratio)
            y = top + 20 if top - 20 < 20 else top - 20
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 3)
            cv2.rectangle(frame, (right, top), (left, bottom),
                          (255, 0, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("w"):
            break

        # reset names every frame
        names = []

    # on exit, close windows and stop stream
    cv2.destroyAllWindows()


if __name__ == "__main__":
    skyhunt("/Users/faizanrasool/Desktop/school/ObjectDetection/combined/encodings.pickle", "cnn")
