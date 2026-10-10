#Bu ödev Muhammed YILMAZ ın öedevidir.


import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO("yolov8n.pt")

# Görüntüyü oku
img = cv2.imread("img.jpg.jpg")

if img is None:
    raise FileNotFoundError(
        "img.jpg bulunamadı! Dosyanın konumunu kontrol et."
    )

results = model(img)
image = img.copy()

# Tespitleri incele(nolur)
for result in results:
    for box in result.boxes:
        conf = float(box.conf[0])

        if conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Merkez koordinatları
            X = int((x1 + x2) / 2)
            Y = int((y1 + y2) / 2)

            # Nesnenin çevresine dikdörtgen çizecek
            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            # Merkez noktasını çizecek
            cv2.circle(
                image, (X, Y), 5,
                (0, 0, 255), -1
            )

            # Koordinatları yazacak
            cv2.putText(
                image, f"({X}, {Y})",
                (X, Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2
            )

# Sonucu kontrol(Heyecanlı kısım!!)
if cv2.imwrite("cikti.jpg", image):
    print("İşlem tamamlandı! cikti.jpg kaydedildi.")
else:
    print("Görüntü kaydedilemedi.")
