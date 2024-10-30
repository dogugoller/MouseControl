import cv2
import mediapipe  # gerçek zamanlı bilgisayarla görme işlemleri için kütüphane
import pyautogui  # fare kontrolü için gerekli kütüphane

camera = cv2.VideoCapture(0)  # varsayılan kamerayı aç

mpHands = mediapipe.solutions.hands  # mediapipeye el takip modulünü tanımla

hands = mpHands.Hands()  # el takibi için gerekli ayarları yap

mpDraw = mediapipe.solutions.drawing_utils  # el işaretlerini çizmeye yarar

# ekran boyutlarını al
screen_width, screen_height = pyautogui.size()

while True:  # kamera kapanana kadar döngüyü devam ettir

    success, img = camera.read()  # görüntü başarıyla alındı mı, alınan görüntü

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mediapipe rgb kabul eder bu yüzden rgbye çevir

    hlms = hands.process(imgRGB)  # hlms değişkeninde görüntüyü takip için işle
    print(hlms.multi_hand_landmarks)

    if hlms.multi_hand_landmarks:
        for handlandmarks in hlms.multi_hand_landmarks:  # tespit edilen eller için döngü
            # işaret parmağı ucunun koordinatlarını al [baş parmak uç kısım ve işaret parmağı uç kısım]
            index_finger_tip = handlandmarks.landmark[8] # işaret parmağı
            thumb_tip = handlandmarks.landmark[4] # başparmak

            # görüntü boyutlarını öğrenip koordinatları yeniden ölçeklendir
            img_height, img_width, _ = img.shape # channel değişkenine ihtiyaç olmadığından , _ koyduk
            x_finger = int(index_finger_tip.x * img_width)
            y_finger = int(index_finger_tip.y * img_height)

            # ekran boyutuna göre koordinatları yeniden ölçeklendir [fare imlecini doğru konuma taşımak için]
            screen_x = int(screen_width * index_finger_tip.x)
            screen_y = int(screen_height * index_finger_tip.y)

            # fareyi hareket ettir
            pyautogui.moveTo(screen_x, screen_y)

            # işaret parmağı ile baş parmak arasındaki mesafe (tıklama kontrolü)
            thumb_x, thumb_y = int(thumb_tip.x * img_width), int(thumb_tip.y * img_height)
            distance = ((x_finger - thumb_x)**2 + (y_finger - thumb_y)**2) ** 0.5 # mesafe

            # mesafe belli bir değerden küçükse tıklama yap [iki elin parmağının yakınlık durumuna göre tıklama işlemi]
            if distance < 50:
                pyautogui.click()

            # tespit edilen işaretlerini görüntü üzerinde çiz
            mpDraw.draw_landmarks(img, handlandmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("El Takip", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # q tuşuna kapanana kadar açık tut [1] → 1 milisaniye
        break

camera.release()
cv2.destroyAllWindows()
