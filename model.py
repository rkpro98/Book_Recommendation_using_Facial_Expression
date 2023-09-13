from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D, Activation,Flatten,Dropout,Dense
from tensorflow.keras import backend as K 
from sklearn.model_selection import train_test_split
import numpy as np
import random
import cv2
import os
import glob
import matplotlib.pyplot as plt
epochs=170
lr=1e-3
batch_size=170
img_dims=(60,60,3)

data=[]
labels=[]
image_files=[f for f in glob.glob(r'C:\Users\rkpro\Desktop\test_folder\test_project_folder\dataset'+"/**/*",recursive=True)if not os.path.isdir(f)]
random.shuffle(image_files)
for img in image_files:
  image=cv2.imread(img)
  image=cv2.resize(image,(img_dims[0],img_dims[1]))
  image=img_to_array(image)
  data.append(image)
  label=img.split(os.path.sep)[-2]
  if label=="angry":
    label=0
  elif label=="happy":
    label =1
  elif label=="neutral":
      label=2  
  elif label=="sad":
      label=3                                       
  labels.append([label])
data=np.array(data,dtype="float")/255.0
labels=np.array(labels)
(trainX, testX, trainY, testY)=train_test_split(data,labels,test_size=0.2,random_state=42) 
trainY=to_categorical(trainY,num_classes=4)  
testY=to_categorical(testY,num_classes=4)
print(np.shape(trainX))
print(np.shape(trainY))
aug=ImageDataGenerator(rotation_range=25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")
def build(width,height,depth,classes):
    model=Sequential()
    inputShape=(height,width,depth)
    chanDim=-1

    if K.image_data_format()=="channels_first":
        inputShape=(depth,height,width)
        chanDim=1

    model.add(Conv2D(40,(5,5),padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3),padding="same"))
    model.add(Dropout(0.25)) 

    model.add(Conv2D(30,(4,4),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3),padding="same"))
    model.add(Dropout(0.25)) 

 


   

    
     


  


    


 





    
    
      


    

    
   
    

    model.add(Flatten())
       
    
    
     
   
    
    
    model.add(Dense(1000))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5)) 

    


    model.add(Dense(250))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5)) 


   

     

    model.add(Dense(4))
    model.add(Activation("softmax"))
    
    return model
model=build(width=img_dims[0],height=img_dims[1],depth=img_dims[2],classes=4)
opt=Adam(lr=lr,decay=lr/epochs)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
history=model.fit_generator(aug.flow(trainX,trainY,batch_size=batch_size),validation_data=(testX,testY),steps_per_epoch=len(trainX) // batch_size,epochs=epochs,verbose=1)
model.summary()
model.save(r'C:\Users\rkpro\Desktop\test_folder\test_project_folder\model.model')
plt.plot(history.history['accuracy'])
plt.title("model accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.plot(history.history['loss'])
plt.title("model loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()



