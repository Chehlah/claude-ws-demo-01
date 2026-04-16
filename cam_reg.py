import cv2


def list_cameras(max_index=5):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def main():
    cameras = list_cameras()
    if not cameras:
        print("ไม่พบกล้องที่เชื่อมต่อ")
        return

    print(f"พบกล้อง {len(cameras)} ตัว: index {cameras}")

    if len(cameras) > 1:
        print("เลือกกล้องที่ต้องการใช้:")
        for i, idx in enumerate(cameras):
            print(f"  [{i}] Camera index {idx}")
        choice = int(input("กรอกหมายเลข: "))
        cam_index = cameras[choice]
    else:
        cam_index = cameras[0]

    cap = cv2.VideoCapture(cam_index)
    print(f"เปิดกล้อง index {cam_index} — กด 's' เพื่อบันทึกภาพ, 'q' เพื่อออก")

    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            break

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            filename = f"capture_{saved_count:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"บันทึกภาพ: {filename}")
            saved_count += 1
        elif key == ord("q"):
            print("ออกจากโปรแกรม")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
