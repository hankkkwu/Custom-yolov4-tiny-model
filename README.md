# Train a custom yolov4-tiny model and convert the .weight file to .tflite file

Here is the link to [Colab](https://colab.research.google.com/drive/1gMpQaLjLPomF0yklDnJcqb-3wygJfhiS?usp=sharing)

## Training the custom yolov4-tiny model:
For training a custom yolov4-tiny model, we'll use the [darknet](https://github.com/AlexeyAB/darknet) repository.

### Step1: Prepare the custom dataset
#### Method1: Manually collect data and label it
* Collect images and split into `obj` file and `test` file 
* Using [labelImg](https://github.com/tzutalin/labelImg) to label the images. It will create .txt-file for each .jpg-file in the same directory and with the same name.

    For example for img1.jpg you will be created img1.txt containing: 

    object_class | x_center | y_center | width | height

    1 | 0.716797 | 0.395833 | 0.216406 | 0.147222

    0 | 0.687109 | 0.379167 | 0.255469 | 0.158333

* (Optional) Do data augmentation using [Roboflow](https://app.roboflow.ai)

* Create file `train.txt` with filenames of your images, each filename in new line, for example containing:
```
data/obj/img1.jpg
data/obj/img2.jpg
data/obj/img3.jpg
```

* Create file `test.txt` with filenames of your images, each filename in new line, for example containing:
```
data/test/img1.jpg
data/test/img1.jpg
data/test/img1.jpg
```

#### Method2: Using Google's Open Images Dataset
Gathering a dataset from Google's Open Images Dataset and using OIDv4 toolkit to generate labels is easy and time efficient. The dataset contains labeled images for over 600 classes!  [Explore the Dataset Here!](https://storage.googleapis.com/openimages/web/index.html)

* git clone the [OIDv4 toolkit!](https://github.com/theAIGuysCode/OIDv4_ToolKit)

* cd to `OIDv4_ToolKit`

* Create a virtual environment

    `python3 -m venv data_collection`

* Activate the virtual environment

    `source data_collection/bin/activate`

* Install the requirement dependences

    `pip install -r requirements.txt`

* Download the data from Open Images Dataset for training dataset (in this case, I downloaded the following 7 classes: Motorcycle, Bicycle, Car, Bus, Person, Stop_sign, Traffic_light)

    `python3 main.py downloader --classes Motorcycle Bicycle Car Bus Person Stop_sign Traffic_light --type_csv train --limit 500 --multiclasses 1`

* Rename the training dataset folder to train

    `mv OID/Dataset/validation/Motorcycle_Bicycle_Car_Bus_Person_Stop\ sign_Traffic\ light/ OID/Dataset/train/train`

* Download the data from Open Images Dataset for validation dataset

    `python3 main.py downloader --classes Motorcycle Bicycle Car Bus Person Stop_sign Traffic_light --type_csv validation --limit 100 --multiclasses 1`

* Rename the validation dataset folder to test

    `mv OID/Dataset/validation/Motorcycle_Bicycle_Car_Bus_Person_Stop\ sign_Traffic\ light/ OID/Dataset/validation/test`

* Edit the `/OIDv4_ToolKit/classes.txt` with objects names. (each in new line)

* Generate `.txt` file for each image

    `python3 convert_annotations.py`

* Delete the old labels for training

    `rm -r OID/Dataset/train/obj/Label/`

* Delete the old labels for validation

    `rm -r OID/Dataset/validation/test/Label/`

* Compress the `obj` and `test` folder into `obj.zip` and `test.zip`

* Create file `train.txt` with filenames of your images, each filename in new line, for example containing:
```
data/obj/img1.jpg
data/obj/img2.jpg
data/obj/img3.jpg
```

* Create file `test.txt` with filenames of your images, each filename in new line, for example containing:
```
data/test/img1.jpg
data/test/img1.jpg
data/test/img1.jpg
```


### Step2: Train a custom model on darknet on google colab
#### Configure our GPU environment on Google Colab
* check that Nvidia CUDA drivers are already pre-installed and which version is it. This can be helpful for debugging

    `!/usr/local/cuda/bin/nvcc --version`

* see what kind of GPU we have

    `!nvidia-smi`

* Change the number depending on what GPU is listed above, under NVIDIA-SMI > Name
```
# Tesla K80: 30
# Tesla P100: 60
# Tesla T4: 75
%env compute_capability=30
```

#### Install the Darknet YOLOv4 training environment
* clone darknet repo

    `!git clone https://github.com/AlexeyAB/darknet`

* change makefile to have GPU and OPENCV enabled
```
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```

* make darknet (builds darknet so that you can then use the darknet executable file to run or train object detectors)
    
    `!make`


#### Configuring Files for Training
i) cfg file:
* Create file `yolov4-tiny-obj.cfg` with the same content as in `yolov4-tiny-custom.cfg`
* change line batch to `batch=64`
* change line subdivisions to `subdivisions=16`
* change line max_batches to (classes*2000 but not less than number of training images, but not less than number of training images and not less than 6000), i.e. `max_batches=6000` if you train for 3 classes
* change line steps to 80% and 90% of max_batches, i.e. `steps=4800,5400`
* set network size `width=416`, `height=416` or any value multiple of 32
* change line `classes=80` to your number of objects in each of 2 [yolo]-layers
* change [filters=255] to `filters=(classes + 5)x3` in the 2 [convolutional] before each [yolo] layer. (keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers.)

* Copy the custom config file to cfg folder
    `!cp ../mydrive/My\ Drive/yolov4/yolov4-tiny-obj.cfg ./cfg`

ii) obj.names:
* Create file `obj.names` with objects names. (each in new line)
* Copy the `obj.names` file to data folder
    `!cp ../mydrive/My\ Drive/yolov4/obj.names ./data`

iii) obj.data:
* Create file `obj.data` containing (where classes = number of objects):
```
classes = 7
train  = data/train.txt
valid  = data/test.txt
names = data/obj.names
backup = /mydrive/My\ Drive/yolov4/backup    # (Create a backup folder in your google drive and put its correct path in this file.)
```
* Copy the `obj.data` file to data folder
    `!cp ../mydrive/My\ Drive/yolov4/obj.data ./data`


#### Upload our custom dataset for YOLOv4-tiny
* Get access to google drive
```
from google.colab import drive    
drive.mount('/content/mydrive')
```

* copy over datasets into the root directory of the Colab VM (comment out test.zip if you are not using a validation dataset)
```
!cp ../mydrive/My\ Drive/yolov4/obj.zip ../    
!cp ../mydrive/My\ Drive/yolov4/test.zip ../
```

* unzip the datasets and their contents so that they are now in /darknet/data/ folder
```
!unzip ../obj.zip -d data/
!unzip ../test.zip -d data/
```

* copy over `train.txt` and `test.txt` into the ./data directory of the Colab VM
```
!cp ../mydrive/My\ Drive/yolov4/train.txt ./data
!cp ../mydrive/My\ Drive/yolov4/test.txt  ./data
```
* Download pre-trained weights for the convolutional layers

    `!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29`


#### Train our custom YOLOv4-tiny object detector
TIP: This training could take several hours depending on how many iterations you chose in the .cfg file. You will want to let this run as you sleep or go to work for the day, etc. However, Colab Cloud Service kicks you off it's VMs if you are idle for too long (30-90 mins).

To avoid this hold (CTRL + SHIFT + i) at the same time to open up the inspector view on your browser.

Paste the following code into your console window and hit Enter:    
```
function ClickConnect(){
console.log("Working");   
document
    .querySelector('#top-toolbar > colab-connect-button')
    .shadowRoot.querySelector('#connect')
    .click() 
}
setInterval(ClickConnect,60000)
```

* train your custom detector! (uncomment `%%capture` below if you run into memory issues or your Colab is crashing)
Note: `-dont_show` flag stops chart from popping up since Colab Notebook can't open images on the spot, `-map` flag overlays mean average precision on chart to see how accuracy of your model is, only add map flag if you have a validation dataset.
```
%%capture
!./darknet detector train data/obj.data cfg/yolov4-tiny-obj.cfg yolov4-tiny.conv.29 -dont_show -map
```

* defien a `imShow()` function to show chart.png of how custom object detector did with training
```
def imShow(path):
    import cv2
    import matplotlib.pyplot as plt
    %matplotlib inline
    image = cv2.imread(path)
    height, width = image.shape[:2]
    resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
    fig = plt.gcf()
    fig.set_size_inches(18, 10)

    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()
imShow('chart.png')
```

* We can kick off training from where it last saved

    `!./darknet detector train data/obj.data cfg/yolov4-obj.cfg /mydrive/My\ Drive/yolov4/backup/yolov4-obj_last.weights -dont_show`



### Step3: Checking the Mean Average Precision (mAP) of Your Model
Run the following command on any of the saved weights from the training to see the mAP value for that specific weight's file. I would suggest to run it on multiple of the saved weights to compare and find the weights with the highest mAP as that is the most accurate one!

NOTE: If you think your final weights file has overfitted then it is important to run these mAP commands to see if one of the previously saved weights is a more accurate model for your classes.

    `!./darknet detector map data/obj.data cfg/yolov4-tiny-obj.cfg /content/darknet/backup/yolov4-tiny-obj_best.weights`


### Step4: Run Your Custom Object Detector!!!
* need to set our custom cfg to test mode
``` 
%cd cfg
!sed -i 's/batch=64/batch=1/' yolov4-tiny-obj.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov4-tiny-obj.cfg
%cd ..
```

* run your custom detector with this command (upload an image to your google drive to test, thresh flag sets accuracy that detection must be in order to show it)
```
!./darknet detector test data/obj.data cfg/yolov4-tiny-obj.cfg /content/darknet/backup/yolov4-tiny-obj_best.weights ../mydrive/My\ Drive/yolov4/3.jpg -thresh 0.3
imShow('predictions.jpg')
```

---

## Convert the weights to TensorFlow Lite
For converting the .weight file to .tflite file, using this [respository](https://github.com/hunglc007/tensorflow-yolov4-tflite)


### Step1: Install
* clone the repo
```
%cd /content
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git
%cd /content/tensorflow-yolov4-tflite
```

### Step2: Configure
* Change the labels from the default COCO to our own custom ones
```
!cp /content/mydrive/My\ Drive/yolov4/obj.names /content/tensorflow-yolov4-tflite/data/classes/
!ls /content/tensorflow-yolov4-tflite/data/classes/
!sed -i "s/coco.names/obj.names/g" /content/tensorflow-yolov4-tflite/core/config.py
```

### Step3: Upload the .weights file
* Upload the custom .weight file for converting
```
!cp /content/dmydrive/My\ Drive/yolov4/yolov4-tiny-obj_best.weights /content/darknet/backup/
!ls /content/darknet/backup/
```

### Step4: Convert to .pb file
* Convert to TensorFlow SavedModel
```
!python save_model.py \
  --weights /content/darknet/backup/yolov4-tiny-obj_best.weights \
  --output ./checkpoints/yolov4-tiny-pretflite-416 \
  --input_size 416 \
  --model yolov4 \
  --tiny \
  --framework tflite
```

### Step5: Convert to TensorFlow Lite
* From the generated TensorFlow SavedModel, we will convert to .tflite
```
%cd /content/tensorflow-yolov4-tflite
!python convert_tflite.py --weights ./checkpoints/yolov4-tiny-pretflite-416 --output ./checkpoints/yolov4-tiny-416.tflite
```

### Step6: Save your Model
* You can save your model to your Google Drive for further use.
```
!cp -r /content/tensorflow-yolov4-tflite/checkpoints/yolov4-tiny-416/ "/content/mydrive/My Drive/yolov4"
!cp /content/tensorflow-yolov4-tflite/checkpoints/yolov4-tiny-416.tflite "/content/mydrive/My Drive/yolov4"
```