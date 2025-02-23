import cv2

face_detection_model_path = 'write path to haarcade_frontalface'

class Facedetector:
    def __init__(self,model_path= face_detection_model_path):
        """
        Cunstructor loads the cascade face detection classifier.
        """
        self.cascade = cv2.CascadeClasifier(model_path)
        self.face = []

    def detect_face(self,frame):
        '''
        Datect faces in the given frame.
        :param frame: Input image(numpy array)
        '''
        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.equlizeHist(gray_image,gray_image)

        # Detectin face 
        self.face = self.cascade.detecMultiScale(
            gray_image,scaleFacator = 1.1 , minNeigbors = 2,
            flags = cv2.CASCADE_SCALE_IMAGE,minSize=(100,100)
        )
    
    def draw_bounding_box_on_frame(self,frame,image_obj):
        '''
        Draws boundaries boxes around deted faces.
        :param frame: Input image (numpy array)
        :patram image_obj: Image object to store results
        :return: Update Image object with boundaries boxes
        '''
        if len(self.face)>0:
            for (x,y,w,h) in self.faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h), (250,0,0),3) # X and Y position on the picture and w and h rectangular damensions X&Y will be top corner on face (X+W,Y+H) represent bottom_right boundaries of the box of face
                roi_image = frame[y:y + h, x:x + w]  # Selecting all rowas and all collumns ,( h represent variable hight and w represet variable width(filtration))
                image_obj.set_roi(roi_image) # extracted face stored in image object 
                image_obj.set_frame(frame) # extracted frame stored in image object 
        
        return image_obj
    
    def print_prediction_text_to_frame(self,image_obj,emotion_predictions):
        """
        Prints predicted emotions on detected faces.
        :param image_obj: Image object containing the frame
        :param emotion_predictions: List of predicted emotions
        :return: Updated Image object with text annotations
        """
        img = image_obj.get_frame()
        if len(image_obj)>0:
            for i,(x,y,w,h) in enumerate(self.faces):
                cv2.putText(img,emotion_predictions[i],(x,y-10),cv2.FONT_HERSHEY_DUPLEX,1.0,(118,185,0),2) # emotion prediction will be located top right corner -10
        
        image_obj.set_frame(img) # we update frames in image object with img 

        return image_obj
