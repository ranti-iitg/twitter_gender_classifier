import urllib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from keras.utils import np_utils
from sklearn.utils import shuffle

twigen = pd.read_csv("gender-classifier-DFE-791531.csv", encoding='latin1')
twigen.head()
'''
#to download data
for imgurl in twigen['profileimage']:
	fname=twigen['profileimage'][i].split('/')[-1]
	im=cv2.imread("./img/"+fname)
	urllib.urlretrieve(imgurl, "./img/"+filename)
	if im==None:
	
'''
'''
imlist=[]
genderlist=[]
target = open("result.txt", 'w')
for i in xrange(len(twigen['profileimage'])):
        unit_id=twigen['_unit_id'][i]
	fname=twigen['profileimage'][i].split('/')[-1]
	im=cv2.imread("./img/"+fname)
	if im==None:
	        target.write(str(unit_id))
	        target.write(" , ")
	        target.write(str(0.33))
	        target.write(",")
	        target.write(str(0.33))
	        target.write(",")
	        target.write(str(0.33))
	        target.write("\n")
	else:
	        res=cv2.resize(im,(50,50))
	        res=res/255.0
	        res2=res.reshape((1,50,50,3))
	        p=model.predict(res2)
	        target.write(str(unit_id))
	        target.write(" , ")
	        target.write(str(p[0][0]))
	        target.write(",")
	        target.write(str(p[0][1]))
	        target.write(",")
	        target.write(str(p[0][2]))
	        target.write("\n")

for i in xrange(len(twigen['profileimage'])):
	fname=twigen['profileimage'][i].split('/')[-1]
	im=cv2.imread("./img/"+fname)
	gender=twigen['gender'][i]
	if im!=None and (gender=='male' or gender=='female' or gender=='brand') :
		res=cv2.resize(im,(50,50))
		imlist.append(res)
		genderlist.append(gender)

imlistnp=np.array(imlist)
genderlistnp=np.array(genderlist)

np.savez('gender_data.npz',
        imgs=imlistnp,
        gender=genderlistnp)
'''
#from collections import Counter
#Counter(list)
#reading data

all_data = np.load('gender_data.npz')
imgs_color = all_data['imgs']
imgs_color=imgs_color/255.0
gender_list= all_data['gender']
imgs_color_male=[]
imgs_color_female=[]
imgs_color_brand=[]

#np_gender_list_y=np.ones(len(gender_list))
for i in xrange(len(gender_list)):
	if gender_list[i]=='male':
		imgs_color_male.append(imgs_color[i])
	elif gender_list[i]=='female':
		imgs_color_female.append(imgs_color[i])
	elif gender_list[i]=='brand':
		imgs_color_brand.append(imgs_color[i])
	else:
		print "exception"


imgs_color_male_np=np.array(imgs_color_male)
imgs_color_female_np=np.array(imgs_color_female)
imgs_color_brand_np=np.array(imgs_color_brand[:1700])
imgs_color_np=np.vstack((imgs_color_male_np,imgs_color_female_np,imgs_color_brand_np))
list_zeros=[0]*imgs_color_male_np.shape[0]
list_ones=[1]*imgs_color_female_np.shape[0]
list_twos=[2]*imgs_color_brand_np.shape[0]
list_all=list_zeros+list_ones+list_twos
np_gender_list_y=np.array(list_all)
np_final_gender_list_y = np_utils.to_categorical(np_gender_list_y)
#training
X,Y=shuffle(imgs_color_np,np_final_gender_list_y)
#example directly taken from keras.io
#from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.applications.vgg19 import VGG19

# create the base pre-trained model
base_model = VGG19(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
print imgs_color.shape
print np_final_gender_list_y.shape

h = model.fit(X,Y, verbose=1, validation_split=0.1, nb_epoch=10,shuffle=True)
model.save("my_weight.h5")


from keras.models import load_model
model = load_model('my_model.h5')
''''
if you want more training
In [68]: for layer in model.layers[:18]:
    ...:    layer.trainable = False
    ...: for layer in model.layers[18:]:
    ...:    layer.trainable = True
    ...:    
'''''
