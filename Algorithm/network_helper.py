import os
import logging as log
from openvino.inference_engine import IECore, IENetwork


class Network:

    def load_network_to_IE(self, model_xml, device):
        """
        Initialisiert die Inferenz Enginge
        Parameter
        -----------
        model_xml : str
            Pfad zu der xml-Architektur der IR
        device : str
            Name des Gerätes auf der die Inferenz ausgeführt werden soll
            z.B. MYRIAD 
        """
        self.core = IECore()
        self.device = device
        self.network = self.core.read_network(model=model_xml, weights=os.path.splitext(model_xml)[0] + ".bin")
        self.core.load_network(self.network, device)
        self.__initialize_input_output_blob()
    
    
    def get_input_shape(self):
        """
        Liefert die Form des Input Layers
        """
        return self.network.input_info['image'].input_data.shape


    def initialize_executable_network(self):
        """
        Lädtd das im Kern definierte Netzwerk und erstellt ein ausführbare Netzwerk
        auf dem Inferenz durchgeführt werden kann.
        """
        self.exec_network = self.core.load_network(self.network, self.device)  
    
    def async_request(self, image):
        """
        Führt eine asynchrone Inferenz auf der IR durch und wartet bis diese fertig ist.
        
        Parameter
        -----------
        image : numpy.ndarray
            Ein bereits transponiertes Bild, welches für die Inferenz genutzt werden kann. 
        """
        request = self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        request.wait()
    
    def get_network_result(self):
        """
        Gib das Ergebnis der zuvor ausgeführten Inferenz durch async_request() zurück.
        """
        return self.exec_network.requests[0].output_blobs[self.output_blob].buffer

    def __initialize_input_output_blob(self):
        self.input_blob = self.__get_input_blob()
        self.output_blob = self.__get_output_blob()

    def __get_input_blob(self):
        return next(iter(self.network.input_info))
    
    def __get_output_blob(self):
        return next(iter(self.network.outputs))
