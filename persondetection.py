import numpy as np
import tensorflow as tf
import time
import os
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior() #Disables tesnorflow 2.x

class DetectorAPI:
    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__)) #Gets current path 
        self.path_to_ckpt = "frozen_inference_graph.pb" #Path of model
        self.detection_graph = tf.Graph() 

        with self.detection_graph.as_default(): #Making detection graph as default
            od_graph_def = tf.GraphDef() #Initializing graph defination
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph) #Parsing
                tf.import_graph_def(od_graph_def, name='') #Importing the parsed model

        self.default_graph = self.detection_graph.as_default() #Storing default graph
        self.sess = tf.Session(graph=self.detection_graph) #Creating tensorflow session

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0') #Input tesnor
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0') #Output tensor
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        image_np_expanded = np.expand_dims(image, axis=0) #Expanding the image
        start_time = time.time() #Starting timer
        (boxes, scores, classes, num) = self.sess.run( #Running model on a frame and getting all outputs
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time() #Ending timer

        im_height, im_width, _ = image.shape #
        boxes_list = [None for i in range(boxes.shape[1])] #Storing ccordinates of boxes

        for i in range(boxes.shape[1]): #Coverting into pixel
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),int(boxes[0, i, 1]*im_width),int(boxes[0, i, 2] * im_height),int(boxes[0, i, 3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self): #Clossing session and Graph
        self.sess.close()
        self.default_graph.close()