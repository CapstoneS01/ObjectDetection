import imutils.video
import pickle
import cv2
import face_recognition
from djitellopy import Tello
import face_recognition
import time
import torch


class TelloYolo:
    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)

    def load_model(self):
        model = torch.hub.load(
            'yolov5', 'custom', path='yolov5/yolov5s.pt', source='local')
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(
                    row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(
                    labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame


def skyhunt(encodings_path, detection_method):
    # object detection
    detection = TelloYolo()

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

        # object detection
        results = detection.score_frame(frame)
        frame = detection.plot_boxes(results, frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("w"):
            break

        # reset names every frame
        names = []

    # on exit, close windows and stop stream
    cv2.destroyAllWindows()


if __name__ == "__main__":
    skyhunt("combined/encodings.pickle", "hog")
