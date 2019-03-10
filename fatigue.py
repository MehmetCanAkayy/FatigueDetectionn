

dosya='/home/mehmetcan/PycharmProjects/untitled4/2-FemaleNoGlasses.avi'
from scipy.spatial import distance as dist
from timeit import default_timer as timer



import argparse
import dlib
import numpy as np
import cv2
import pandas as pd
import time
import timeit



def t_time():
    start = timer()
    time.sleep(1)
    end = timer()
    return end-start
#timer ile gözün ne zamanlar arasında kapalı ve açık olduğunu gösteren fonksiyon.

counter_time=0
count=0
#Göz kırpamanın algılanması ve sayılması için aşağıdaki tanımlamalar yapılmaktadır.
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
	help="path to input video file")
ap.add_argument("-t", "--threshold", type = float, default=0.27,
	help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type = int, default=2,
	help="the number of consecutive frames the eye must be below the threshold")

EYE_AR_THRES = 0.3
EYE_AR_CONSEC_FRAME = 48

COUNTER = 0
args = vars(ap.parse_args())
EYE_AR_THRESH = args['threshold']
list_lenLimit = 150
EAR_Tol = []
screenshot_cnt = 0
durum=0
TOTAL=0
degiskenDosya = open('uyku2.osd', 'w')
degiskenDosya1= open('esneme2.osd', 'w')
Sayilar = []
Sayilar1 = []

say=0
sonuc=0
liste_say = len(Sayilar)
EYE_AR_CONSEC_FRAMES = args['frames']



def goz_enboy_orani(eye):
    #goz-enboy_orani olarak tanımlanan fonksiyonuna parametre olarak yüz üzerindeki göze
    # ait noktaların atandığı dizi olan eye dizisi gönderilir.
    #‘a’ ile tanımlanan kısımda eye[1],eye[5] yani yüz üzerindeki 38-42. noktalar arasındaki
    # öklid uzaklığı ve ‘b’ ile de 39-41 arasındaki mesafe hesaplanır.
    #Daha sonra ‘c’ de 37-40 arasındaki yatay mesafe hesaplanarak ear ile tanımlanan değişkende bu değerler işleme alınır.

    # oklid mesafesini hesapliyorum
    #gozun koordinatlari
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # yataydaki oklid mesafesi

    c = dist.euclidean(eye[0], eye[3])

    ear = (a + b) / (2 * c)

    return ear
def agiz_acikligi(mouth):
    #Ağız içinde aynı mantıkta ilerleyerek ağız açıklığını bulmak
    # için en üst ve en alt nokta arasındaki uzaklık bulunur.
    a = dist.euclidean(mouth[4], mouth[10])
    mouth=a

    return mouth

def burun_seviyesi(nose): # oklid mesafesini hesapliyorum
    #gozun koordinatlari
    a = dist.euclidean(nose[0],((right_eye[0]+left_eye[0])/2))
    nose=a
    #print(nose)

    return nose
def shape_to_np(shape, dtype="int"):
    # (x, y) koordinatlarinin listesini baslat
    coords = np.zeros((68, 2), dtype=dtype)

    # 68 yuz simgesi uzerinde dongu
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)


    return coords

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Yüz bölgesinin tespit edilmesinde kullanılan kısım predictor. Ve yüz bölgesini işaretliyor..


#kameradan goruntu alma:
#video = cv2.VideoCapture(0)
#videodan goruntu alma:
video= cv2.VideoCapture(dosya)
while(video.isOpened()):
    _, frame = video.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray_frame, 0)
    #Griye çeviriyoruz işlem yapabilmek için.

    for rect in rects:

#Kamera üzerinden denemeler yapıldığından kamera kısmı kod içerisinde aktiftir.videonu n okunması
# ile başlayan döngüde dlib kütüphanesinde yüzün kısımlarına ait olan noktalar isimlendirilmiştir.
# Örneğin sağ göz koordinatları dlib kütüphanesinde 36-42 arasındaki noktalar ile belirtilmektedir.

        shape = predictor(gray_frame, rect)
        shape = shape_to_np(shape)
        #nereyi istiyorsak oraya karsilik gelen koordinatlar
        mouth=shape [48:68]
        left_eye = shape[42: 48]
        right_eye = shape[36: 42]
        nose=shape[27:35]

#Noktaların belirlenmesinin ardından kamera üzerinden alınan görüntüde göz,ağız ve burun
# kısımlarının renklendirilmesi aşağıdaki gibi yapılmaktadır.

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        mouth_hull=cv2.convexHull(mouth)
        nose_hull=cv2.convexHull(nose)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), )
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame,[mouth],-1,(0,0,255),1)
        cv2.drawContours(frame,[nose],-1,(0,0,255),1)




        leftEAR = goz_enboy_orani(left_eye)
        rightEAR = goz_enboy_orani(right_eye)
        mouthEAR=agiz_acikligi(mouth)
        noseEAR=burun_seviyesi(nose)
        avgEAR = (leftEAR + rightEAR) / 2.0
        EAR_Tol.append(avgEAR)

        #print(avgEAR)
        """if(burun_seviyesi>=30):
            cv2.putText(frame, "kafasi asagida", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)"""
        if(mouthEAR>=30):
            durum=1
            counter_time += t_time()
            count = counter_time
            if count >= 3:
                cv2.putText(frame, "saniye: {:.2f}".format(count), (500, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "ESNEME ALGILANDI", (500, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #degiskenDosya1.write('{}'.format(count) + "\n")
                #print(count)
                Sayilar.append(count)
                """for i in range(len(Sayilar)):
                  sonuc= Sayilar[len(Sayilar)-1]-Sayilar[0]"""



            else:
                count=0
                """liste_say = len(Sayilar)
                baslangic = 0
                print(Sayilar[1] - Sayilar[0])"""



        else:
            for i in range(len(Sayilar)):
                sonuc = Sayilar[len(Sayilar) - 1]
            if sonuc>0:
                degiskenDosya1.write('{}'.format(sonuc) + " SANİYE ESNEME ALGILANDI" + "\n")
                #degiskenDosya1.write('{}'.format(sonuc) + "\n")
                counter_time = 0
                # Dosyalama aşamasında açılan bir .osd dosyasına veriler kaydedilmektedir.Bu aşama aşağıdaki gibidir.
                #degiskenDosya1.write("Esnemiyor\n")
                #print('Normal')
                #cv2.putText(frame, "Esnemiyor", (30, 60),
                 #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #degiskenDosya.write("1\n")


        #print (mouthEAR)



#Hesaplamanın ardından en başta belirtilmiş olan değer ile karşılaştırılarak uykulu-uykusuz durumuna karar verilir.
#Ağız açıklığının belirlediğimiz bir değerden yüksek olup olmadığını kontrol ederek esneme durumuna karar verilir.

        if avgEAR < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
               TOTAL+=1
            COUNTER=0
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

         #cv2.putText(frame, "EAR: {:.2f}".format(avgEAR), (300, 30),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


               #degiskenDosya.write("1\n")
        #goz acikliği 0.24den kucuk olacak
        if avgEAR< 0.24:
            COUNTER+=1
            durum=1






        else :
            if COUNTER>= EYE_AR_CONSEC_FRAME:
               COUNTER=0
               durum=0


#Göz açıklığının algılanmasının ardından esneme içinde timer kullanılmıştır.Ağız açıklığının belirli bir düzeyin
# üstüne çıkmasına artı olarak belirli bir süreden fazla o açıklıkta  kalması da esneme
# belirtisini güçlendiren bir etken olarak düşünülerek kodlama kısmı aşağıdaki gibi geliştirilmiştir.

    if(durum==1):
        counter_time += t_time()
        count=counter_time
        if count>=3:
            #cv2.putText(frame, "saniye: {:.2f}".format(count), (500, 30),
             #       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #counter_time=0
            Sayilar1.append(count)

            #degiskenDosya.write('{}'.format(count)+"SANİYE KAPALI KALDI"+"\n")
            #print(count)


            #cv2.putText(frame, "Uykulu", (30, 60),
             #       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else :
            count=0
            #cv2.putText(frame, "saniye: {:.2f}".format(count), (500, 30),
                       # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        for i in range(len(Sayilar1)):
            say = Sayilar1[len(Sayilar1) - 1]
        if say >0:
            degiskenDosya.write('{}'.format(say) + " SANİYE KAPALI KALDI" + "\n")
        counter_time=0
        # Dosyalama aşamasında açılan bir .osd dosyasına veriler kaydedilmektedir.Bu aşama aşağıdaki gibidir.
       # degiskenDosya.write("Uyanık\n")
       # print('Normal')
       # cv2.putText(frame, "UYANIK", (30, 60),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    durum=0
    count=0


    cv2.imshow('cam', frame)

    if len(EAR_Tol) >= list_lenLimit:
        EAR_Tol = EAR_Tol[-list_lenLimit:]

    k = cv2.waitKey(20)
#Uygulamanın kullanılabilirliği açısından ‘q’ tuşuna basıldırğında ekranın kapanması, ‘p’ tuşuna basıldığında durdurulması ve
# ‘s’ tuşuna basıldığında ekran görüntüsünün alınarak dosya dizinine kaydedilmesi sağlanmıştır.
    if k == ord('q') or k == 27:
        break
    elif k == ord('p'):
        cv2.waitKey(0)
    elif k==ord('s') :
        screenshot_cnt+=1
        cv2.imwrite("screenshot_"+str(screenshot_cnt)+".jpg", frame)
degiskenDosya.close()
video.release()
cv2.destroyAllWindows()
#plt.close()



