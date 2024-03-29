import torch
import cv2
from djitellopy import Tello


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

    def __call__(self):
        # tello = Tello()
        # tello.connect()

        # tello.streamoff()
        # tello.streamon()

        while True:
            capture = cv2.VideoCapture(0)
            grabbed, frame = capture.read()

            # frame_read = tello.get_frame_read()
            # frame = frame_read.frame

            if not grabbed:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            cv2.imshow("img", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


detection = TelloYolo()
detection()
