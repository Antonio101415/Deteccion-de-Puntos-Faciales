# Heimdall-EYE Uso:
# python parse_xml.py --input ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml --output ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train_eyes.xml
# python parse_xml.py --input ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml --output ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test_eyes.xml

# Importamos los paquetes necesarios 
import argparse
import re

# Contruimos el analizador de argumentos y analizamos estos argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to iBug 300-W data split XML file")
ap.add_argument("-t", "--output", required=True,
	help="path output data split XML file")
args = vars(ap.parse_args())

# En el cojunto de datos iBUG 300-W cada uno (x,y) se coordina en un mapa especifico
# Caracteristica facial (boca, ojos, cara) 
#En este momento mediante los puntos en el rango 36-48 elegimos los ojos ya que ese rango pertenece a los ojo
LANDMARKS = set(list(range(36, 48)))

# Para analizar facilmente las ubicaciones  de los ojos del archivo XML
# Utilizamos expresiones regulares para determinar si hay una parte 
# Elemento en cualquier linea dada
PART = re.compile("part name='[0-9]+'")

# Cargamos el contenido del archivo XML original y abrir el archivo de salida para la posterior escritura
print("[INFO] parsing data split XML file...")
rows = open(args["input"]).read().strip().split("\n")
output = open(args["output"], "w")

# Recorremos las filas del archivo dividido de datos
for row in rows:
	# Verificamos si la linea actual tiene la coordenadas (x,y) para
	# La referencia de los puntos faciales que nos interesen (Boca , Nariz , Mandibula ...)
	parts = re.findall(PART, row)

	# Si no hay informacion relaccionada con las coordenadas (x,y) de 
	# Los puntos de referencia faciales podemos imprimir la linea actual , sin mas modificaciones
	if len(parts) == 0:
		output.write("{}\n".format(row))

	# De lo contrario hay informacion de anotaciones son las que debemos procesar
	else:
		# Analizamos el nombre del atributo de la fila
		attr = "name='"
		i = row.find(attr)
		j = row.find("'", i + len(attr) + 1)
		name = int(row[i + len(attr):j])

		# Si el nombre del marca facial existe dentro del rango de nuestro indices lo imprimimos en el frame del archivo de salida
		if name in LANDMARKS:
			output.write("{}\n".format(row))

# Cerramos la salida
output.close()
