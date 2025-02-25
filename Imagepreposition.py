import cv2
import numpy as np 


class Image:
    def __init__(self):
        self._frame = None
        self._roi_image = []
        self.model_input_image = []
    
    def get_frame(self):
        """
        Get the current frame.
        :return: The stored frame(numpy array)
        """
        return self._frame
    
    def set_frame(self,frame):
        '''
        Set the current frame.
        :param frame: Input frame(numpy array)
        '''
        self._frame = frame

    def get_roi(self):
        '''
        Add a new ROI to the list.
        :param roi: ROI image(numpy array)
        '''

        return self._roi_image
    
    
    def set_input(self):
        """
        Get the list of processed model input images
        :return: List of model input images 
        """
        return self._model_input_image
    
    def set_roi(self,roi):

        self._roi_image.append(roi)

    def preprocess_roi(self):
        if self._roi_image:
            for roi in self._roi_image:
                # Convert to gratscale 
                gray_image = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

                # Resize the ROI to model input size
                processed_image = cv2.resize(gray_image,(48,48))

                # COnvert image pixels from range 0 - 255 to 0 - 1
                processed_image = processed_image.astype(np.float32)/255.0

                #append to teh model input list 

                self._model_input_image.append(processed_image)






