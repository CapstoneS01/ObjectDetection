from imutils.video import VideoStream
import cv2
import face_recognition
import time
import imutils
import pickle


def video_recognition(encodings_path, detection_method):
    names = []
    # load defined encodings
    data = pickle.loads(open(encodings_path, "rb").read())
    # start webcam stream
    stream = VideoStream(src=0).start()
    time.sleep(2.0)

    while (1):
        # read stream and convert to RGB
        frame = stream.read()
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
    stream.stop()


if __name__ == "__main__":
    video_recognition(
        "face_recognition/encodings.pickle", "hog")
