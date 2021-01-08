import argparse
import numpy as np 
import cv2
import sys
import logging 

from datetime import datetime
from network_helper import Network

IMAGE_DEST_PATH = './Images/'

def args():
    """
    Initialisiert das Setup und die Hilfe, um das Gerät der Inferenz und die Netwerkarchitektur, über die Argumente zu spezifizieren 
    """
    parser = argparse.ArgumentParser(
        description= "Specify model and device."
    )
    parser.add_argument('-d', '--device', help="Specify the device the model will be executed on.[e.g. MYRIAD, CPU, GPU]", default="MYRIAD")
    parser.add_argument('-m', '--model', help="Specify the Path to the .xml file.")
    return parser.parse_args()

def logger_setup():
    """
    Gibt einen initialisierten Logger zurueck, so dass Log Informationen 
    in eine Log-Datei und in der Konsole ausgegeben werden
    """
    logging.basicConfig(filename="patrolSystem.log", format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    return logger

def main():
    logger = logger_setup()
    logger.info("Executing Patrolsystem Algorithm")
    # Hole die übergebenen Argumente und erstelle die Grundlage des Netzwerkes
    arguments = args()
    network = Network()
    # Der IE mitteilen, welches Model und welches Geraet verwendet werden soll
    network.load_network_to_IE(arguments.model, arguments.device)
    logger.info("Initialized IE successfully")
    # Hole die Form des Input-Layers des Netzwerkes
    input_shape = network.get_input_shape()
    # Ein ausführbares Netzwerk erstellen auf dem die Inferenz ausgeführt werden kann
    network.initialize_executable_network()
    logger.info("Initialized executable network succesfully")
    # Stelle eine Verbindung zu der Kamera an Port 0 her (Pi Cam)
    cap = cv2.VideoCapture(0)
    # Erhalte die Hoehe und Breite der Bilder, um die potenziell erkannten Menschen mit Rechtecken zu markieren
    width = int(cap.get(3))
    height = int(cap.get(4)) 
    image_id = 0

    try:
        # Führe die Bildanalyse so lange durch, wie die Verbindung zur Kamera besteht
        logger.info("Analyzing...")
        while cap.isOpened():
            # Erhalte das nächste Bild des Videostreams 
            flag, frame = cap.read()
            # Wenn kein Bild gelesen werden kann stoppe den Algorithmus
            if not flag:
                break
            #Transponiere das geholte Bild so, dass es zu dem Input Layers der IR passt
            b, c, h, w = input_shape
            input_image = cv2.resize(np.copy(frame), (w,h)).transpose((2,0,1)).reshape(1,3,h,w)
            # Führe mit dem transponierten Bild eine asynchrone Inferenz aus
            network.async_request(input_image)
            # Erhalte das Ergebnis der Inferenz
            output = network.get_network_result()
            for box in output[0][0]:
                confidence = box[2]
                # Bei einer Erkennungssicherheit von 60% wird ein Bild gesichert
                if confidence >= 0.6:
                    logger.info("HUMAN DETECTED")
                    xmin, ymin = int(box[3] * width), int(box[4] * height)
                    xmax, ymax = int(box[5] * width), int(box[6] * height)
                    # Zeichne ein rotes Rechteck, um die Person und speichere das Bild im IMAGE_DEST_PATH Ordner
                    img = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                    image_id += 1
                    cv2.imwrite(IMAGE_DEST_PATH + 'DP_Record_'+ str(image_id) +'.jpeg', img)
    # Wenn str + c gedrückt wird, gebe die letzten Ressourcen frei und beende den Algorithmus
    except KeyboardInterrupt:
        cap.release()    
        logger.info("closed the stream successfully")
        
if '__main__' == __name__:
    main()
