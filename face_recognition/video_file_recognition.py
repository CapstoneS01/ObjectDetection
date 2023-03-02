import imutils.video
import face_recognition
import imutils
import pickle
import time
import cv2


def video_file_recognition(encodings_path, input_path, output, method):
    print("[INFO] Loading encodings...")
    data = pickle.loads(open(encodings_path, "rb").read())
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(input_path)

    writer = None
    time.sleep(2.0)

    frame_count = 0
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    while frame_count < total_frames:
        success, frame = vs.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        video_ratio = frame.shape[1] / float(rgb.shape[1])
        face_boxes = face_recognition.face_locations(rgb, model=method)
        encodings = face_recognition.face_encodings(rgb, face_boxes)
        names = []

        for encoding in encodings:

            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)

        for ((top, right, bottom, left), name) in zip(face_boxes, names):
            top = int(top * video_ratio)
            bottom = int(bottom * video_ratio)
            left = int(left * video_ratio)
            right = int(right * video_ratio)
            y = top + 20 if top - 20 < 20 else top - 20
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (255, 0, 0), 3)
            cv2.rectangle(frame, (right, top), (left, bottom),
                          (255, 0, 0), 3)

        if writer is None and output is not None:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter("/Users/faizanrasool/Desktop/school/ObjectDetection/face_recognition/test/" + output, fourcc, 60,
                                     (frame.shape[1], frame.shape[0]), True)

        if writer is not None:
            writer.write(frame)

        frame_count += 1

    cv2.destroyAllWindows()
    vs.release()

    if writer is not None:
        writer.release()


if __name__ == "__main__":
    video_file_recognition("/Users/faizanrasool/Desktop/school/ObjectDetection/face_recognition/encodings.pickle",
                           "/Users/faizanrasool/Desktop/school/ObjectDetection/face_recognition/test/test_video.MOV", "test_output.mp4", "hog")
