from prepare_data import *
from Modele import *
pts_path="300w_train_landmarks.txt"
imgs_path="300w_train_images.txt"
	
pts_test="helen_testset_landmarks.txt"
imgs_test="helen_testset.txt"
m=Model()
#display cropped images
#test_display_cropped(imgs_path,pts_path)
m=train_model(m,imgs_path,pts_path)
m=test_model(m,imgs_test,pts_test)
