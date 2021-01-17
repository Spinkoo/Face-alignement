#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import random
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import pickle
path = 'afw/'


class Model(object):

    """docstring for Model"""

    def __init__(self):
        self.R = []
        self.errors = []
        self.axes = []
        self.mean_face = None

    def computerCentreG(self, x, y):
        (cx, cy) = (np.mean(x), np.mean(y))
        return (cx, cy)

    def loadParams(self):
        Rfile = open('best.r', 'rb')
        axesFile = open('bestAxes.axes', 'rb')
        self.R = pickle.load(Rfile)
        self.axes = pickle.load(axesFile)

    def loadmeanFace(self):
        meanfaceFile = open('meanface.face', 'rb')        
        self.mean_face = pickle.load(meanfaceFile)

    def computeY(self, X):
        Y = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return Y

    def getOptimDepalcemnt(self, pt, pt_m):
        return np.array(pt) - np.array(pt_m)

    def optimDeplacmnts(self, pt, pt_m):
        for i in range(len(pt)):
            pt[i] = self.getOptimDepalcemnt(pt[i], pt_m)
        return pt

    def debugList(self, l):
        print (np.array(l).shape)

    def computeR(self, Y, optim):
        return np.linalg.inv(np.transpose(Y).dot(Y)).dot(np.transpose(Y)).dot(optim)

    def majPoints(self, s, sstar):

        return s + sstar

    def usePCA(self, des):
        pca = PCA(n_components=0.98)
        pca.fit(des)
        X = pca.transform(des)
        return (X, pca.components_)

    def forward(self, Y, R):
        return Y.dot(R)
    def oneStepAlignment(
        self,
        imgs,
        pt,
        pt_mean,
        ):

        sifts = self.computeSiftAll(imgs.copy(), pt_mean.copy())

        s_st = self.getOptimDepalcemnt(pt, pt_mean)
        (X, axes) = self.usePCA(sifts)
        Y = self.computeY(X)

        R = self.computeR(Y, s_st)
        s_0 = self.forward(Y, R)
        err = mean_squared_error(s_0, s_st)
        print("Reduction  from ",sifts.shape," to ",X.shape)
        print ('MSE :', err)
        updated_pts = self.majPoints(pt_mean, s_0)

        im_true = self.draw_both_points(imgs[0].copy(), updated_pts,
        pt, 0)
        cv2.imshow('train', im_true)
        cv2.waitKey(1000)
        self.i+=1
        return (updated_pts, R, axes, err)

    def reshape2D(self, pt):
        f = pt.copy().reshape(pt.shape[0], 68, 2)
        for i in range(pt.shape[0]):
            f[i, :, 0] = pt[i, 0:68]
            f[i, :, 1] = pt[i, 68:136]
        return f

    def computeLocalSift(self, image, keypoints):
        sift = cv2.xfeatures2d.SIFT_create()
        (keypoints, descriptors) = sift.compute(image,
                keypoints=keypoints)
        im = image.copy()
        im = cv2.drawKeypoints(im, keypoints, im)
        return (im, keypoints, descriptors)

    def computeSiftAll(self, images, keypoints):
        flatSifts = []
        print(keypoints.shape)
        keypoints = self.reshape2D(keypoints.copy())
        for (i, img) in enumerate(images):
            keypoint = self.generateKeypoint(keypoints[i].reshape(1,
                    68, 2))
            (ims, _, desc) = self.computeLocalSift(img, keypoint)

            flatSifts.append(desc.reshape(-1))
        return np.array(flatSifts)

    def generateKeypoint(self, pt):
        key_points = []
        for i in range(0, len(pt[0])):
            key_points.append(cv2.KeyPoint(x=pt[0, i, 0], y=pt[0, i,
                              1], _size=20))
        return key_points

    def drawPoints(
        self,
        img,
        ptx,
        pty,
        color=None,
        ):
        img = img.copy()
        for j in range(len(ptx)):
            if color == None:
                img = cv2.circle(img, (int(ptx[j]), int(pty[j])), 2,
                                 (255, 255, 0), 1)
            else:
                img = cv2.circle(img, (int(ptx[j]), int(pty[j])), 2,
                                 color, 1)
        return img

    def fit(
        self,
        imgs,
        landmarks,
        mean_face,
        iters=5,
        ):
        self.i=0
        self.mean_face = mean_face.copy()

        for i in range(iters):

            (pts_new, r, axes, err) = self.oneStepAlignment(imgs,
                    landmarks, self.mean_face)
            print ('Iteration ', i + 1, ' done !')
            self.mean_face = pts_new
            self.R.append(r)
            self.errors.append(err)
            self.axes.append(axes)
        self.mean_face=mean_face
        self.save_params()

    def getBestIndex(self):
        if len(self.errors) == 0:
            return None
        return self.errors.index(min(self.errors))

    def save_params(self):
        i = self.getBestIndex()
        assert not i == None, \
            'Assertion error with fitting : You havent fitted yet '
        Rfile = open('best.r', 'wb')
        axesFile = open('bestAxes.axes', 'wb')
        pickle.dump(self.R[i], Rfile)
        pickle.dump(self.axes[i], axesFile)
    def save_all(self):
        Rfile = open('All.r', 'wb')
        axesFile = open('All.axes', 'wb')
        pickle.dump(self.R, Rfile)
        pickle.dump(self.axes, axesFile)
    def load_all(self):
        Rfile = open('All.r', 'rb')
        axesFile = open('All.axes', 'rb')
        self.R=pickle.load( Rfile)
        self.axes=pickle.load( axesFile)        
    def draw_both_points(
        self,
        img,
        keypoints_alig,
        keypoints_orig,
        index,
        ):
        keypoints = self.reshape2D(keypoints_orig)
        img = self.drawPoints(img.copy(), keypoints[index, :, 0],
                              keypoints[index, :, 1],(0,255,0))  # red are detected ones
        temp = self.reshape2D(keypoints_alig)
        im_true = self.drawPoints(img, temp[index, :, 0], temp[index, :
                                  , 1], (0, 0, 255))  # green points are ground truth
        return im_true
    def test_model_compute(self,R1,A1,imgs,test_pts,index=0):


        X_test = self.computeSiftAll(imgs, self.mean_face.copy())
        X_test1 = X_test.dot(A1.T)  # pca reduction
        Y = self.computeY(X_test1)
        s_test1 = self.forward(Y,R1)  # deplacmnts output
        self.mean_face+=s_test1
        keypoints = self.mean_face 

        im_true = self.draw_both_points(imgs[3].copy(), keypoints,
        test_pts, 3)
        cv2.imwrite(str(index)+'___test.png', im_true)
        cv2.imshow('test', im_true)
        cv2.waitKey(1000)
        return s_test1
    def test_model(
        self,
        imgs,
        X_test,
        pre_trained=False,
        ):
        test_pts = X_test.copy()
        s_test = 0
        if(self.mean_face.shape[0]==136):
            self.mean_face=np.array([self.mean_face]*X_test.shape[0])

        for i in range(len(self.R)):

            R1 = self.R[i]
            A1 = self.axes[i]
            s_star = test_pts - self.mean_face
            s_test1=self.test_model_compute(R1,A1,imgs,test_pts,i)
            s_test += s_test1
            print ('Iteration ', str(i + 1))
            print(mean_squared_error(s_test1,s_star))
            if pre_trained:
                break
