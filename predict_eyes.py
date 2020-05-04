# Heimdall-EYE USO:
# python predict_eyes.py --shape-predictor eye_predictor.dat

# Importamos los paquetes o librerias necesarios
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

# Contruimos el analizador de argumentos y analizamos estos argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# Inizializamos el detector de cara de dlib (basado en HOG) y luego cargamos nuestro predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Inicializamos el detector de la camara 
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Recorremos los fotogramas de la trasmision de video
while True:
	# Tomamos el fotograma de la transmision de video , cambiamos el tamaño
	# Para tener un ancho maximo de 400 pixeles y convertilo en una escala de grises
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detectamos rosotros faciales en una escala de grises
	rects = detector(gray, 0)

	# Realizamos un bucle para la deteccion de rostros faciales
	for rect in rects:
		# Convertimos el rectangulo dlib en un cuadro delimitador con OPENCV y 
		# Dibujamos un cuadro delimitador que rodea el contorno de la cara
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# Usamos nuestro predictor de dlib personalizado para predecir la ubicacion
		# De nuestras coordenadas historicas y luego la convertimos esta prediccion
		# A una matriz Numpy facilmente analizable
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# Bucle sobre las coordenadas (x,y) de nuestra forma dlib
		# Modelo predictor dibujarlos en la imagen
		for (sX, sY) in shape:
			cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)

	# Mostramos el frame en la pantalla 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Si queremos salir del bucle solo basta con pulsar q y saldremos del Frame
	if key == ord("q"):
		break

# Realizamos una limpieza
cv2.destroyAllWindows()
vs.stop()
