# Heimdall-EYE USO:
# python train_shape_predictor.py --training ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train_eyes.xml --model eye_predictor.dat

# Importamos los paquetes necesarios
import multiprocessing
import argparse
import dlib

# Contruimos el analizador de argumentos y analizamos esos argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to input training XML file")
ap.add_argument("-m", "--model", required=True,
	help="path serialized dlib shape predictor model")
args = vars(ap.parse_args())

# Tome las opciones predeterminadas para el predictor de forma de dlib
print("[INFO] setting shape predictor options...")
options = dlib.shape_predictor_training_options()

# Define la profundidad de cada arbol de regresion habra un total de
# 2 elevado hojas de profundidad de arbol en cada arbol pequeños valores de tree_depth
# Sera mas rapido pero menos preciso 
# Mientras que los valores de los mas grande genera arboles mas profundos y a la su vez mas precisos , pero que se ejecutaran mucho mas lento para realizar la predicciones
options.tree_depth = 4

# Parametro para la regularizacion en el rango [0,1] que se utiliza para ayudar
# Nuestro modelo garantiza: valores mas cercanos a 1 haran que nuestro modelo se ajuste
# Los datos de entrenamiento son mejores pero podrian causar un sobreajuste en los mas cercanos 
# A 0 ayudara a nuestro modelo a generalizar pero requirira que rengamos datos de entrenamiento en el orden de miles de puntos de datos
options.nu = 0.1

# La cantidad de cascadas utilizadas para entrenar el predictor de forma esto
# Parametro tiene un impacto dramatico en la precision y salida
# Tamaño de su modelo , Cuantas mas cascadas tenga mas preciso
# Su modelo potencialmente pero tambien mas grande el tamaño de salida
options.cascade_depth = 15

# Numero de pixeles utilizados para generar entidades para los arboles aleatorios en cada cascada : los valores de pixeles mas grande haran que su forma
# Sea mas predictiva mas preciso pero a su vez mas lento use valores grandes si la velocidad no es un problema 
# En nuestro caso a utilizar una raspberry es un problema por lo que utilizamos valores pequeños para recursos bajos de GPU 
options.feature_pool_size = 400

# Selecciona las mejores funciones en cada cascada cuando se entrena cuanto mas grande este valor
# Mas tardara en entrenar pero mas preciso sera nuestra prediccion
options.num_test_splits = 50

# Controla la cantidad de "jitter" (Es decir el aumento de datos) durante el entrenamiento 
# El predictor de forma que aplica un numero proporcionado de aleatorio 
# Deformaciones realizando asi la regularizacion y aumento la capacidad de garantizar nuestro modelo
options.oversampling_amount = 5

# Cantidad de jitter de traduccion para aplicar los documentos dlib recomiendan unos valores entre [0,0.5]
options.oversampling_translation_jitter = 0.1

# Le decimos al predicot de forma dlib que sea detallado y imprima el estado 
options.be_verbose = True

# Numero de subprocesos / nucleos de CPU que se utilizaran durante el entrenamiento por defecto
# Este valor cogera los nucleos que el sistema tenga pero podemos dar un valor enetero si lo queremos 
options.num_threads = multiprocessing.cpu_count()

# Registramos nuestras opciones de entrenamiento en el terminal
print("[INFO] shape predictor options:")
print(options)

# Comenzamos con el entrenamiento de prediccion
print("[INFO] training shape predictor...")
dlib.train_shape_predictor(args["training"], args["model"], options)
