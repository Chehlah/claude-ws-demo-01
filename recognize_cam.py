import csv
import cv2
import sqlite3
import numpy as np
import torch
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ─── Config ───────────────────────────────────────────────────────────────────
DB_PATH    = "faces.db"
IMG_SIZE   = 160
THRESHOLD  = 0.9

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"ใช้ device: {DEVICE}")

# ─── Models ───────────────────────────────────────────────────────────────────
# MTCNN บน CPU (MPS ไม่รองรับ adaptive_avg_pool2d)
mtcnn = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    keep_all=True,
    post_process=True,
    device=torch.device("cpu"),
)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


# ─── Database ─────────────────────────────────────────────────────────────────
def load_records(db_path: str = DB_PATH) -> list[dict]:
    """โหลดข้อมูลนักเรียนพร้อม embedding จาก table students"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT name, student_no, year, embedding FROM students"
    ).fetchall()
    conn.close()

    records = []
    for name, student_no, year, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        records.append({
            "name":       name,
            "student_no": student_no,
            "year":       year,
            "embedding":  vec,
        })
    return records


# ─── Cosine distance ──────────────────────────────────────────────────────────
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a, b))


def identify(embedding: np.ndarray, records: list[dict]) -> tuple[dict | None, float]:
    """คืน (record ที่ใกล้ที่สุด, distance) หรือ (None, 1.0) ถ้าไม่จำได้"""
    best_rec, best_dist = None, THRESHOLD
    for rec in records:
        dist = cosine_distance(embedding, rec["embedding"])
        if dist < best_dist:
            best_dist = dist
            best_rec  = rec
    return best_rec, best_dist


# ─── Export CSV ───────────────────────────────────────────────────────────────
def export_csv(seen: dict[str, dict], period: str) -> str:
    """
    seen: { dedup_key → record + seen_at }
    เรียงตาม student_no แล้ว export เป็น CSV
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"attendance_{period}_{timestamp}.csv"

    rows = sorted(seen.values(), key=lambda r: r["student_no"])
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["student_no", "name", "year", "period", "seen_at"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "student_no": row["student_no"],
                "name":       row["name"],
                "year":       row["year"],
                "period":     period,
                "seen_at":    row["seen_at"],
            })
    return filename


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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    records = load_records()
    if not records:
        print("ยังไม่มีข้อมูลใน DB กรุณารัน register_cam02.py ก่อน")
        return
    print(f"โหลด {len(records)} นักเรียนจาก DB")

    period = input("กรอกคาบที่ต้องการเช็กชื่อ: ").strip()

    cap = open_camera()
    print(f"Threshold: {THRESHOLD} | กด 'q' เพื่อ export CSV และออก")

    # dedup key = student_no + year ป้องกันนับซ้ำ
    seen: dict[str, dict] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            break

        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(img_rgb)

        boxes, _     = mtcnn.detect(pil_img)
        face_tensors = mtcnn(pil_img)

        if boxes is not None and face_tensors is not None:
            for box, face_tensor in zip(boxes, face_tensors):
                x1, y1, x2, y2 = map(int, box)

                t = face_tensor.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = facenet(t).squeeze().cpu().numpy()

                rec, dist = identify(emb, records)

                if rec is not None:
                    dedup_key = f"{rec['student_no']}_{rec['year']}"
                    if dedup_key not in seen:
                        seen[dedup_key] = {**rec, "seen_at": datetime.now().isoformat()}
                        print(
                            f"เช็กชื่อ: {rec['name']} "
                            f"(เลขที่ {rec['student_no']} ชั้น {rec['year']}) "
                            f"dist={dist:.3f}"
                        )

                    label = f"{rec['name']} ({dist:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown ({dist:.2f})"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                )

        # แสดงจำนวนคนที่เช็กชื่อแล้วมุมบนซ้าย
        cv2.putText(
            frame, f"Checked: {len(seen)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if seen:
        filename = export_csv(seen, period)
        print(f"\nExport สำเร็จ → {filename}")
        print(f"เช็กชื่อทั้งหมด {len(seen)} คน")
    else:
        print("ไม่มีข้อมูลการเช็กชื่อ")


if __name__ == "__main__":
    main()
