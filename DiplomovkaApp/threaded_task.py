import threading
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob
import time
import visualization_utils as vis_util

from utils import label_map_util

start = time.time()
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


class ThreadedTask(threading.Thread):

    def __init__(self, progressbar,state_text,images_folder,save_folder,
                 selected_model,working_text, sort_var,hightlight_var):
        threading.Thread.__init__(self)
        self.progressbar = progressbar
        self.state_text = state_text
        self.images_folder = images_folder
        self.save_folder = save_folder
        self.selected_model = selected_model
        self.working_text = working_text
        self.sort_var = sort_var
        self.hightlight_var = hightlight_var

    def run(self):
        # Name of the directory containing the object detection module we're using
        CWD_PATH = os.getcwd()
        if self.selected_model == 1:
            MODEL_NAME = os.path.join(CWD_PATH,"models/faster_rcnn_inception_v2_coco/inference_graph")
        if self.selected_model == 2:
            MODEL_NAME = os.path.join(CWD_PATH,"models/faster_rcnn_resnet101_coco/inference_graph")
        if self.selected_model == 3:
            MODEL_NAME = os.path.join(CWD_PATH,"models/faster_rcnn_inception_resnet_v2_atrous_coco/inference_graph")
        if self.selected_model == 4:
            MODEL_NAME = os.path.join(CWD_PATH,"models/faster_rcnn_resnet50_coco/inference_graph")


        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(os.getcwd(), 'labelmap.pbtxt')

        # Path to image
        PATH_TO_IMAGE = self.images_folder.get()

        PATH_TO_SAVE_IMAGES_DIR = self.save_folder.get()

        # Number of classes the object detector can identify
        NUM_CLASSES = 5

        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        #PATH_TO_TEST_IMAGES_DIR = '/Users/matusko/models/research/object_detection/testimg/'
        # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
        #print(PATH_TO_IMAGE)
        data_path = os.path.join(PATH_TO_IMAGE, '*g')
        TEST_IMAGE_PATHS = glob.glob(data_path)
        self.progressbar.config(mode="determinate", maximum=len(TEST_IMAGE_PATHS) + 1, value=1)
        self.progressbar.place(x=50, y=450)
        i = 1
        for image_path in TEST_IMAGE_PATHS:

            # Load image using OpenCV and
            # expand image dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            image = cv2.imread(image_path)
            image_expanded = np.expand_dims(image, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            if(self.sort_var ==0):
                self.hightlight_var = 1

            # Draw the results of the detection (aka 'visulaize the results')
            image, classes_array= vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                self.hightlight_var,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.80)
            #print(triedy_pole)

            if(self.sort_var ==1):
                if len(classes_array) == 0:
                    path = os.path.join(PATH_TO_SAVE_IMAGES_DIR, "nedetegovane")
                    if os.path.exists(path):
                        path = os.path.join(PATH_TO_SAVE_IMAGES_DIR, "nedetegovane", 'image{}.jpg'.format(i))
                        cv2.imwrite(path, image)
                    else:
                        os.makedirs(path)
                        path = os.path.join(PATH_TO_SAVE_IMAGES_DIR, "nedetegovane", 'image{}.jpg'.format(i))
                        cv2.imwrite(path, image)

                else:
                    for trieda in classes_array:
                        path = os.path.join(PATH_TO_SAVE_IMAGES_DIR,trieda)
                        if os.path.exists(path):
                            path = os.path.join(PATH_TO_SAVE_IMAGES_DIR, trieda, 'image{}.jpg'.format(i))
                            cv2.imwrite(path, image)
                        else:
                            os.makedirs(path)
                            path = os.path.join(PATH_TO_SAVE_IMAGES_DIR, trieda, 'image{}.jpg'.format(i))
                            cv2.imwrite(path, image)
            else:

                path = os.path.join(PATH_TO_SAVE_IMAGES_DIR, 'image{}.jpg'.format(i))
                cv2.imwrite(path, image)

            i = i + 1
            self.state_text.set(str(i - 1) + "/" + str(len(TEST_IMAGE_PATHS)))
            self.progressbar.step(1)  # Update progress bar

        self.working_text.set("                       ---D O K O N C E N E---                       ")

        print('The script took {0} second !'.format(time.time() - start))