import cv2
from ultralytics import YOLO


def main():
    model = YOLO("bestfix.pt")

    # Ganti sesuai hasil tes kamera
    CAMERA_INDEX = 0

    # Membuka webcam eksternal
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    # Periksa apakah kamera berhasil dibuka
    if not cap.isOpened():
        print("Kamera gagal dibuka menggunakan DirectShow.")

        # Coba backend alternatif Windows
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("Kamera tidak bisa dibuka!")
        return

    # Atur resolusi setelah kamera berhasil dibuka
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Kamera berhasil dibuka!")
    print("Resolusi:",
          cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          "x",
          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Tekan Q untuk keluar.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        # Resize frame
        frame = cv2.resize(
            frame,
            (1280, 720),
            interpolation=cv2.INTER_AREA
        )

        # YOLO Detection
        results = model.predict(frame, conf=0.5, verbose=False)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLO External Camera", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()