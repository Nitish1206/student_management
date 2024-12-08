import pickle
import cv2
import dlib
import tensorflow as tf
from face_reco.face_utils import *
import time
from numpy import dot
from numpy.linalg import norm
from mtcnn import MTCNN
import numpy as np
import os
# import tensorflow.keras.backend as K

import sys
from threading import Thread
from datetime import datetime
from queue import Queue
import traceback


class Face_recogntion():
    def __init__(self):
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session()
            with self.sess.as_default():
                self.face_graph = tf.compat.v1.GraphDef()
                fid = tf.compat.v1.compat.v2.io.gfile.GFile("face_reco\Models\\facenet.pb", 'rb')
                serialized_graph = fid.read()
                self.face_graph.ParseFromString(serialized_graph)
                tf.compat.v1.import_graph_def(self.face_graph, name='')
                self.facenet_sess = tf.compat.v1.compat.v1.Session(graph=self.graph)
                self.images_placeholder = self.graph.get_tensor_by_name("input:0")
                self.embeddings = self.graph.get_tensor_by_name("embeddings:0")
                self.detector = MTCNN()

        self.msg_box_timeout = 5  # in secs
        self.accuracy = 0
        self.predictor_68 = dlib.shape_predictor("face_reco\Models\shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor("face_reco\Models\shape_predictor_5_face_landmarks.dat")
        self.embeddings_file = "face_reco\Models\\recognition.pickle"

        if os.path.exists(self.embeddings_file):
            self.user_data = pickle.loads(open(self.embeddings_file, 'rb').read())
        else:
            self.user_data = []
        # print("user data...",self.user_data)
        # print(self.user_data)
        self.threshold = 0.75
        self.face_training = False
        self.desired_left_eye = (0.35, 0.35)
        self.desired_face_width = 160
        self.desired_face_height = None
        if self.desired_face_height is None:
            self.desired_face_height = self.desired_face_width
        self.width, self.height = 149, 149
        self.face_training = False
        self.divider = 2

        self.faces = [0, 1]
        self.recog_image = self.recog_shape = None
        self.detection_thread = True

        self.frame_queue = Queue(maxsize=1)
        self.name = "New Person"
        self.is_new_person = False
        # self.recognition_thread = Thread(target=self.detect_and_recognize)
        # self.recognition_thread.start()
        self.recognition_status=False

    def training(self, frame, gray, faces, name):
        """Record new person data"""
        data = {}
        print("len",len(faces))
        if len(faces) == 1:
            x, y, w, h = self.get_single_face_rect()
            x = max(x, 0)
            y = max(y, 0)
            shape = self.get_shape_predictor(gray, dlib.rectangle(x, y, x + w, y + h))
            aligned_face = self.align(frame, shape)
            aligned_face = cv2.resize(aligned_face, (160, 160))
            embeddings = self.get_embeddings(aligned_face)
            data["name"] = name
            data["embeddings"] = embeddings[0]
            # self.user_data.append(data)
            # self.save_embeddings()
            return ["success", data["embeddings"]]
        elif len(faces) == 0:
            # print("No face found")
            return ["failed",None]
        else:
            # print("Multiple faces fond! Please try again")
            return ["failed",None]


    def detection(self, small_frame):
        open_time = time.time()
        self.rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        with self.graph.as_default():
            with self.sess.as_default():
                self.faces = self.detector.detect_faces(self.rgb)
                detector_time = time.time() - open_time
                # print("Detector time", detector_time,flush=True)

    def get_single_face_rect(self):
        width = 0
        final_face = None
        for face in self.faces:
            if face["box"][2] > width:
                width = face["box"][2]
                final_face = face["box"]
        return final_face

    def recognition(self, small_frame):
        open_time = time.time()
        if len(self.faces) == 1:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = self.get_single_face_rect()
            # self.detect_status = True
            self.shape = self.get_shape_predictor(gray, dlib.rectangle(x, y, x + w, y + h))
            shape_time_inc = time.time() - open_time
            # print("Shape pred time inc.", shape_time_inc,flush=True)
            st = time.time()
            name, self.accuracy = self.recognize_person(small_frame, self.shape)
            # print("Temp Name", name, self.accuracy, flush=True)
            recognition_time = time.time() - open_time
            # print("Recognition time inc.", recognition_time,flush=True)
            if self.accuracy > self.threshold:
                return "success" , name
            else:
                return "failed",name

    def detect_and_recognize(self,small_frame):

        self.detection(small_frame)
        result,name=self.recognition(small_frame)
        return result,name

    def save_embeddings(self):
        f1 = open(self.embeddings_file, "wb")
        pickle.dump(self.user_data, f1)
        f1.close()


    def get_embeddings(self, aligned_face):
        prewhitened = self.prewhiten(aligned_face)
        feed_dict = {self.images_placeholder: [prewhitened]}
        embeddings = self.facenet_sess.run(self.embeddings, feed_dict=feed_dict)
        return embeddings


    def get_shape_predictor(self, gray, rect):
        shape = self.predictor(gray, rect)
        shape = self.shape_to_np(shape)
        return shape


    @staticmethod
    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def find_similarity(self, emb):
        """Find Similarity for known person"""
        name = "New Person"
        max_value = 0
        similarities = list(map(lambda x: dot(x["embeddings"], emb[0]) / (norm(x['embeddings']) * norm(emb[0])),
                                self.user_data))
        if len(similarities) > 0:
            max_value = max(similarities)
            max_similarity_pos = similarities.index(max_value)
            name = self.user_data[max_similarity_pos]["name"]
        return name, max_value


    def recognize_person(self, image, shape):
        # print("got into recognition ===>>>>>>>>")
        with self.graph.as_default():
            with self.sess.as_default():
                aligned = self.align(image, shape)
                aligned = cv2.resize(aligned, (160, 160))
                embeddings = self.get_embeddings(aligned)
                name, accuracy = self.find_similarity(embeddings)
        return name, accuracy

    def align(self, image, shape):
        # print("align")
        # print("shape",shape)
        # convert the landmark (x, y)-coordinates to a NumPy array
        left_eye_pts = shape[0:2]
        right_eye_pts = shape[2:4]
        # compute the center of mass for each eye
        left_eye_center = left_eye_pts.mean(axis=0).astype("int")
        right_eye_center = right_eye_pts.mean(axis=0).astype("int")
        # compute the angle between the eye centroids
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = 180 - np.degrees(np.arctan2(dY, dX))
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
        desired_dist *= self.desired_face_width
        scale = desired_dist / dist
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = (int(left_eye_center[0] + right_eye_center[0]) // 2,
                       int(left_eye_center[1] + right_eye_center[1]) // 2)

        # print("center==", eyes_center)
        # print("angle==", angle)
        # print("scale==", scale)
        # print("center0",eyes_center[0],type(eyes_center[0]),type(eyes_center))
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, -angle, scale)

        # update the translation component of the matrix
        tx = self.desired_face_width * 0.5
        ty = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (tx - eyes_center[0])
        M[1, 2] += (ty - eyes_center[1])

        # apply the affine transformation
        (w, h) = (self.desired_face_width, self.desired_face_height)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)
        # return the aligned face
        # print("aligned")
        return output


    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords


    def rect_to_bb(self, rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return x, y, w, h

