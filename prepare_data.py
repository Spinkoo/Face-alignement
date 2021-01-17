from glob import glob
import cv2
import numpy as np
import math
import random
from tp import *
import os


def getPts(pts):
	ptx=[]
	pty=[]
	f=open(pts,"r")
	f=f.readlines()
	for j in f :
		j=j.replace("\n","")
		j=j.split(" ")
		ptx.append((float(j[0])))
		pty.append((float(j[1])))
	return ptx,pty

def displayData(i_path,p_path):
	pts=p_path
	imgs=i_path
	nb_samples=len(pts)
	for i in range(20):
		image=cv2.imread(imgs[i])

		ptx,pty=getPts(pts[i])
		minx,miny,maxx,maxy=getBBox(ptx,pty)
		image=drawPoints(image,ptx,pty)
		image = cv2.rectangle(image, (minx,miny), (maxx,maxy), (255,255,0), 2) 
	
		cv2.imshow("",image)
		cv2.waitKey(1000)
def displayCroppedData(imgs,keypoints):
	for i in range(len(imgs)):
		image=imgs[i].copy()

		image=drawPoints(image,keypoints[i,:,0],keypoints[i,:,1])
	
		cv2.imshow("",image)
		cv2.waitKey(1000)
def generate_pert(ptx,pty):
    tran = np.round((np.random.rand(2,10)-0.5)*20)
    scale = (np.random.rand(2,10)-0.5) * 0.2
    ptsx=[]
    ptsy=[]
    for i in range(10):
        ptx = ptx - tran[0,i]
        pty = pty - tran[1,i]      
        ptx = ptx * (scale[0,i] + 1)
        pty = pty * (scale[1,i] + 1)
        ptsx.append(ptx)
        ptsy.append(pty)
    return ptsx,ptsy
def generateKeypoint(ptx,pty):
	key_points=[]
	for i in range (0,len(ptx)):
		key_points.append(cv2.KeyPoint(x=ptx[i],y=pty[i],_size=20))
	return key_points
def getBBox(ptx,pty):
		minx=math.floor(min(ptx))
		miny=math.floor(min(pty))
		maxx=math.ceil(max(ptx))
		maxy=math.ceil(max(pty))
		return minx,miny,maxx,maxy

def cropData(i_path,p_path,to_save=False):
	images=[]
	pointsx=[]
	pointsy=[]
	pts=p_path
	imgs=i_path
	nb_samples=len(pts)

	for i in range(nb_samples):
		
		pts[i]=pts[i].replace("\\","/")
		if not os.path.exists(imgs[i]):
			continue
		image=cv2.imread(imgs[i])
		ptx,pty=getPts(pts[i])

		ptx=np.array(ptx)
		pty=np.array(pty)
		minx,miny,maxx,maxy=getBBox(ptx,pty)
		Lx=int(0.15*(maxx-minx))
		Ly=int(0.15*(maxy-miny))
		ptx=ptx-minx+Lx
		pty=pty-miny+Ly
		factorx=128/(maxx-minx+2*Lx)
		factory=128/(maxy-miny+2*Ly)
		ptx*=factorx
		pty*=factory

		cropped=image[max(miny-Ly,0):maxy+Ly,max(minx-Lx,0):maxx+Lx]


		resized = cv2.resize(cropped, (128,128), interpolation = cv2.INTER_AREA)

		images.append(resized.copy())
		resized=drawPoints(resized,ptx,pty)
		pointsx.append(ptx)
		pointsy.append(pty)
		imgs[i]=imgs[i].replace("\\","/")
		path_save_imsg=imgs[i].split("/")
		#cv2.imshow("",resized)
		#cv2.waitKey(100)

	return images,pointsx,pointsy

def getMeanFace(ptxx,ptyy):
	ptx,pty=ptxx.copy(),ptyy.copy()
	mean_x,mean_y=np.mean(ptx,axis=0),np.mean(pty,axis=0)
	sumx=np.zeros((68,),np.float)
	sumy=np.zeros((68,),np.float)

	x,y=generate_pert(mean_x.copy(),mean_y.copy())
	ptx.extend(x)
	pty.extend(y)
	return np.mean(ptx,axis=0),np.mean(pty,axis=0)

def drawPoints(img,ptx,pty,color=None):
	for j in range(len(ptx)):
		if(color==None):
			img = cv2.circle(img,(int(ptx[j]),int(pty[j])) , 2 ,(255,255,0), 1)
		else:
			img = cv2.circle(img,(int(ptx[j]),int(pty[j])) , 2 ,color, 1)

	return img

		

def computeLocalSift(image,keypoints):
	sift = cv2.xfeatures2d.SIFT_create()
	keypoints, descriptors= sift.compute(image,keypoints=keypoints)
	im=image.copy()
	im=cv2.drawKeypoints(im, keypoints,im)
	return im,keypoints,descriptors
def computeSiftAll(images,keypoints):
	flatSifts=[]
	for i in images:
		_,_,desc=computeLocalSift(i,keypoints)
		flatSifts.append(desc.reshape(-1))
	return np.array(flatSifts)
def combineLists(ptx,pty):
	pt=[]
	for i,item in enumerate(ptx):
		a=list(ptx[i])
		a.extend(list(pty[i]))
		pt.append(a)
	return pt
def combineElems(ptx,pty):
	pt=[]
	for i in ptx:
		pt.append(i)
	for i in pty:
		pt.append(i)
	return pt
def get_paths(path_imgs,path_pts):
	f=open(path_imgs,"r")
	f=f.readlines()
	imgs=[]
	for j in f :
		j=j.replace("\n","")
		imgs.append(j)
	f=open(path_pts,"r")
	f=f.readlines()
	pts=[]
	for j in f :
		j=j.replace("\n","")
		pts.append(j)

	return imgs,pts
def reshape2D(pt):
	f = pt.copy().reshape(pt.shape[0], 68, 2)
	for i in range(pt.shape[0]):
		f[i, :, 0] = pt[i, 0:68]
		f[i, :, 1] = pt[i, 68:136]
	return f
def prepare_train_data(imgs_path,pts_path):
	i_path,p_path = get_paths(imgs_path,pts_path)

	imgs,ptx,pty=cropData(i_path,p_path)
	meanx,meany=getMeanFace(ptx,pty)
	pt=combineLists(ptx,pty)
	pt_mean=combineElems(meanx,meany)
	pt_mean= [pt_mean] * len(ptx)
	return np.array(pt),np.array(pt_mean),imgs
def prepare_test_data(imgs_path,pts_path):
	i_path,p_path = get_paths(imgs_path,pts_path)
	imgs,ptx,pty=cropData(i_path,p_path)

	pt=combineLists(ptx,pty)
	return np.array(pt),imgs
def train_model(m,train_imgs_path,train_pts_path):
	landmarks,pt_mean,imgs=prepare_train_data(train_imgs_path,train_pts_path)
	meanfaceFile = open('meanface.face', 'wb')
	pickle.dump(np.mean(pt_mean, axis=0), meanfaceFile)
	m.fit(imgs,landmarks,pt_mean)
	m.save_all()
	dump_data(landmarks,"landmarks.L")
	dump_data(imgs,"imgs.I")
	return m
def test_model(m,test_imgs_path,test_pts_path,pre_trained=False):
	if pre_trained:
		m.loadParams()
	m.load_all()
	m.loadmeanFace()
	landmarks,pt_mean,imgs=prepare_train_data(test_imgs_path,test_pts_path)
	m.test_model(imgs,landmarks,pre_trained)
	return m
def dump_data(data,name):
	file_save_dump=open(name,'wb')

	pickle.dump(data,file_save_dump)
def load_data(name):
	f = open(name,"rb")
	data=pickle.load(f)
	return data
#display first few images
def test_display_cropped(imgs_path,pts_path):	
	landmarks,pt_mean,imgs=prepare_train_data(imgs_path,pts_path)

	displayCroppedData(imgs[0:10],reshape2D(landmarks[0:10]))

