import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

# Class Deteksi Tangan
class HandDetection:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mpHands.Hands(max_num_hands=max_num_hands, 
                                   min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)

    def findHandLandMarks(self, image, draw=False):
        originalImage = image.copy()
        # Ubah ke RGB sesuai requirement mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        handDataList = []

        # Jika tangan terdeteksi
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Dapatkan label tangan (kanan/kiri)
                hand_label = results.multi_handedness[hand_idx].classification[0].label

                landMarkList = []
                imgH, imgW, imgC = originalImage.shape
                for id, landMark in enumerate(hand_landmarks.landmark):
                    xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                    landMarkList.append([id, xPos, yPos])

                # Tambahkan informasi tangan dan landmark ke list
                handDataList.append({'label': hand_label, 'landmarks': landMarkList})

                # Gambar landmark dan koneksi tangan jika 'draw=True'
                if draw:
                    mpDraw.draw_landmarks(originalImage, hand_landmarks, mpHands.HAND_CONNECTIONS)
                    # Tampilkan label tangan di atas gambar
                    cv2.putText(originalImage, hand_label, 
                                (landMarkList[0][1] - 20, landMarkList[0][2] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return handDataList, originalImage

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

# Inisialisasi HandDetection
handDetection = HandDetection()

while True:
    success, frame = cap.read()
    if not success:
        break

    # Membalik frame secara horizontal (mirror effect)
    frame = cv2.flip(frame, 1)

    # Deteksi dan tampilkan landmark tangan
    handsData, annotatedImage = handDetection.findHandLandMarks(image=frame, draw=True)

    # Tampilkan gambar dengan deteksi tangan
    cv2.imshow("Hand Detection", annotatedImage)

    # Keluar dari loop jika 'x' ditekan
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
