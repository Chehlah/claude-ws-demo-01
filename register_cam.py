import cv2
import sqlite3
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
DB_PATH = "faces.db"
IMG_SIZE = 160

# Apple MPS > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"ใช้ device: {DEVICE}")

# ─── Models ───────────────────────────────────────────────────────────────────
# MTCNN ใช้ CPU เสมอ — MPS ไม่รองรับ adaptive_avg_pool2d ที่ MTCNN ใช้ภายใน
mtcnn = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    keep_all=False,
    post_process=True,
    device=torch.device("cpu"),
)
# FaceNet ใช้ MPS/CUDA/CPU ตามที่มี
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


# ─── Database ─────────────────────────────────────────────────────────────────
def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            student_no  TEXT    NOT NULL,
            year        TEXT    NOT NULL,
            embedding   BLOB    NOT NULL,
            registered_at TEXT  NOT NULL
        )
    """)
    conn.commit()
    return conn


def already_registered(conn: sqlite3.Connection, student_no: str) -> bool:
    row = conn.execute(
        "SELECT id FROM students WHERE student_no = ?", (student_no,)
    ).fetchone()
    return row is not None


def save_student(
    conn: sqlite3.Connection,
    name: str,
    student_no: str,
    year: str,
    embedding: np.ndarray,
):
    blob = embedding.astype(np.float32).tobytes()
    conn.execute(
        """INSERT INTO students
           (name, student_no, year, embedding, registered_at)
           VALUES (?, ?, ?, ?, ?)""",
        (name, student_no, year, blob, datetime.now().isoformat()),
    )
    conn.commit()


# ─── Face pipeline ────────────────────────────────────────────────────────────
def detect_and_embed(frame_bgr: np.ndarray) -> np.ndarray | None:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    face_tensor = mtcnn(pil_img)
    if face_tensor is None:
        return None

    face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = facenet(face_tensor)

    return embedding.squeeze().cpu().numpy()


# ─── Camera helpers ───────────────────────────────────────────────────────────
def list_cameras(max_index: int = 5) -> list[int]:
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def open_camera() -> cv2.VideoCapture:
    cameras = list_cameras()
    if not cameras:
        raise RuntimeError("ไม่พบกล้องที่เชื่อมต่อ")

    if len(cameras) > 1:
        print(f"พบกล้อง {len(cameras)} ตัว:")
        for i, idx in enumerate(cameras):
            print(f"  [{i}] Camera index {idx}")
        choice = int(input("เลือกกล้อง: "))
        cam_index = cameras[choice]
    else:
        cam_index = cameras[0]

    cap = cv2.VideoCapture(cam_index)
    print(f"เปิดกล้อง index {cam_index}")
    return cap


# ─── Input helpers ────────────────────────────────────────────────────────────
def prompt_year() -> str:
    year = input("ชั้นปี (ใช้ทุกคนในรอบนี้) : ").strip()
    if not year:
        raise ValueError("กรุณากรอกชั้นปี")
    return year


def prompt_metadata(year: str) -> dict:
    print("\n─── ข้อมูลนักเรียน ───────────────────────")
    name       = input("ชื่อ-สกุล : ").strip()
    student_no = input("เลขที่     : ").strip()

    if not all([name, student_no]):
        raise ValueError("กรุณากรอกข้อมูลให้ครบทุกช่อง")

    return {"name": name, "student_no": student_no, "year": year}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    conn = init_db()

    try:
        year = prompt_year()
    except ValueError as e:
        print(e)
        conn.close()
        return

    cap = open_camera()
    meta = None
    print("\nกด SPACE เพื่อถ่ายภาพและลงทะเบียน | กด 'n' เพื่อกรอกข้อมูลคนถัดไป | กด 'q' เพื่อออก")

    while True:
        if meta is None:
            try:
                meta = prompt_metadata(year)
            except ValueError as e:
                print(e)
                continue

            if already_registered(conn, meta["student_no"]):
                print(
                    f"เลขที่ {meta['student_no']} ลงทะเบียนแล้ว "
                    "กรุณากรอกข้อมูลใหม่"
                )
                meta = None
                continue

        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            break

        preview = frame.copy()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(Image.fromarray(img_rgb))

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                preview, "Face detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
            )
        else:
            cv2.putText(
                preview, "No face", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2,
            )

        info = f"{meta['name']}  no.{meta['student_no']}  yr.{meta['year']}"
        cv2.putText(
            preview, info, (10, preview.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1,
        )

        cv2.imshow("Student Registration", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            embedding = detect_and_embed(frame)
            if embedding is None:
                print("ไม่พบใบหน้าในภาพ กรุณาลองใหม่")
            else:
                save_student(conn, **meta, embedding=embedding)
                print(
                    f"\nลงทะเบียนสำเร็จ\n"
                    f"  ชื่อ-สกุล : {meta['name']}\n"
                    f"  เลขที่    : {meta['student_no']}\n"
                    f"  ชั้นปี    : {meta['year']}\n"
                    f"  embedding : {embedding.shape}\n"
                )
                ans = input("บันทึกข้อมูลคนถัดไป? (y/n): ").strip().lower()
                if ans == "y":
                    meta = None
                else:
                    print("ออกจากโปรแกรม")
                    break
        elif key == ord("n"):
            meta = None
        elif key == ord("q"):
            print("ออกจากโปรแกรม")
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()


if __name__ == "__main__":
    main()
