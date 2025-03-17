import cv2 
import numpy as np 

class Model:
    def __init__(self, model_filename):
        """
        Initialize the Model class by loading a pretrained Tensorflow model.
        :param model_filename: PAth to the model file (.pb format)
        """
        self.network = cv2.dnn.readNet(model_filename)
        self.classid_to_string = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprised",
            6: "Neutral"
        } # Mapping class IDS to emotion labels 

    def predict (self, image):
        """
        Run model inference on the provided Image object.
        :param image: An instance of hte Image class containing ROI images
        :return: List of predicted emotions with confidence scores 
        """
        roi_iamges = image.get_model_input() # get preprocessed regions of interest
        emotions_predictions = []

        if roi_images:
            for roi in roi_images:
                # convert to blob format 
                blob = cv2.dnn.blobFromImage(roi)

                # pass blob through the network 
                self.network.setInput(blob)
                prob = self.network.forward 