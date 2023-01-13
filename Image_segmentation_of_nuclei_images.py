#%%
# Import packages
import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

import tensorflow as tf
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras import callbacks

#%%
# Data loading
# Define path of the dataset
PATH = os.path.join(os.getcwd(), 'Dataset', 'data-science-bowl-2018-2')

#%%
# Define train and test path
train_path = os.path.join(PATH, 'train')
test_path = os.path.join(PATH, 'test')

# Define the inputs and masks path for both dataset
train_input_path = os.path.join(train_path, 'inputs')
train_mask_path = os.path.join(train_path, 'masks')
test_input_path = os.path.join(test_path, 'inputs')
test_mask_path = os.path.join(test_path, 'masks')

#%%     
# Create empty list for train and test images
train_inputs = []
train_masks = []
test_inputs = []
test_masks = []

# Load the images for train data
train_input_dir = os.path.join(train_path,'inputs')
train_masks_dir = os.path.join(train_path,'masks')
for mask_file in os.listdir(train_masks_dir):
    train_mask = cv2.imread(os.path.join(train_masks_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    train_mask = cv2.resize(train_mask,(128,128))
    train_masks.append(train_mask)
for input_file in os.listdir(train_input_dir):
    train_img = cv2.imread(os.path.join(train_input_dir,input_file))
    train_img = cv2.cvtColor(train_img,cv2.COLOR_BGR2RGB)
    train_img = cv2.resize(train_img,(128,128))
    train_inputs.append(train_img)

# Load the images for test data
test_input_dir = os.path.join(test_path,'inputs')
test_masks_dir = os.path.join(test_path,'masks')
for mask_file in os.listdir(test_masks_dir):
    test_mask = cv2.imread(os.path.join(test_masks_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    test_mask = cv2.resize(test_mask,(128,128))
    test_masks.append(test_mask)
for input_file in os.listdir(test_input_dir):
    test_img = cv2.imread(os.path.join(test_input_dir,input_file))
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img,(128,128))
    test_inputs.append(test_img)

#%%
# Convert to numpy array
train_inputs = np.array(train_inputs)
train_masks = np.array(train_masks)
test_inputs = np.array(test_inputs)
test_masks = np.array(test_masks)

#%%
# Data preprocessing
# Expand the mask dimension
train_masks = np.expand_dims(train_masks,axis=-1)
test_masks = np.expand_dims(test_masks,axis=-1)

#%%
# Convert the mask values from [0,255] into [0,1]
train_masks = np.round(train_masks / 255.0).astype(np.int64)
test_masks = np.round(test_masks / 255.0).astype(np.int64)

#Check the mask output
print(np.unique(train_masks[0]))
print(np.unique(test_masks[0]))

#%%
#3.3. Normalize the inputs
train_inputs = train_inputs/ 255.0
test_inputs = test_inputs/ 255.0

#%%
print(np.unique(train_inputs[0]))
print(np.unique(test_inputs[0]))

#%%
# Convert the loaded images and masks from numpy array to tensor slices
train_inputs_tensor = tf.data.Dataset.from_tensor_slices(train_inputs)
train_masks_tensor = tf.data.Dataset.from_tensor_slices(train_masks)
test_inputs_tensor = tf.data.Dataset.from_tensor_slices(test_inputs)
test_masks_tensor = tf.data.Dataset.from_tensor_slices(test_masks)

# %%
# Combine the images and masks using the zip method
train_dataset = tf.data.Dataset.zip((train_inputs_tensor, train_masks_tensor))
test_dataset = tf.data.Dataset.zip((test_inputs_tensor, test_masks_tensor))

# %%
# Define data augmentation pipeline as a single layer through subclassing
class Augment(keras.layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = keras.layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_masks = keras.layers.RandomFlip(mode='horizontal',seed=seed)

    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_masks(labels)
        return inputs, labels

#%%
# Build the dataset
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEP_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_dataset.batch(BATCH_SIZE)

# %%
# Visualize some pictures as example
def display(display_list):
    plt.figure(figsize=(20,20))
    title = ['Input Image','True Mask','Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
    plt.show()

for inputs,masks in train_batches.take(2):
    sample_image,sample_mask = inputs[0],masks[0]
    display([sample_image,sample_mask])

# %%
# Model development
# Use MobileNetV2 pretrained model as feature extractor
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
base_model.summary()

# Use these activation layers as the outputs from the feature extractor
layer_names = [
    'block_1_expand_relu',      #64x64
    'block_3_expand_relu',      #32x32
    'block_6_expand_relu',      #16x16
    'block_13_expand_relu',     #8x8
    'block_16_project'          #4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Instantiate the feature extractor
down_stack = keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack.trainable = False

# Define the upsampling path
up_stack = [
    pix2pix.upsample(512,3),    #4x4 --> 8x8
    pix2pix.upsample(256,3),    #8x8 --> 16x16
    pix2pix.upsample(128,3),    #16x16 --> 32x32
    pix2pix.upsample(64,3)      #32x32 --> 64x64
]

# Use functional API to construct U-net
def unet(output_channels:int):
    inputs = keras.layers.Input(shape=[128,128,3])
    #Downsample through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    # Build the upsampling path and establish the concatenation
    for up, skip in zip(up_stack,skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x,skip])
    #Use a transpose convolution layer to perform the last upsampling and as output layer
    last = keras.layers.Conv2DTranspose(filters=output_channels,kernel_size=3,  strides=2,padding='same')
    outputs = last(x)
    model = keras.Model(inputs=inputs,outputs=outputs)
    return model

#%%
# Use the function created for model creation
OUTPUT_CHANNELS = 3
model = unet(OUTPUT_CHANNELS)
model.summary()
keras.utils.plot_model(model)

# %%
# Model compilation
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

#%%
# Function to show prediction
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

show_predictions()

#%%
# Callback function
class DisplayCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch {}\n'.format(epoch+1))

#%%
# Tensorboard and earlystopping callbacks
import datetime
from tensorflow.keras import callbacks
log_path = os.path.join('Image_segmentation_nuclei_images',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir=log_path)
es = keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,verbose=1,restore_best_weights=True)

#%%
# Model training
EPOCHS = 40
VALIDATION_SUBSPLITS = 5
VALIDATION_STEPS = 200// BATCH_SIZE 
history = model.fit(train_batches, validation_data=test_batches, epochs=EPOCHS,steps_per_epoch=STEP_PER_EPOCH, validation_steps=VALIDATION_STEPS, callbacks=[DisplayCallback(), tb, es])

# %%
# Model deployment
show_predictions(test_batches,3)

#%%
# Final model evaluation
test_loss,test_acc = model.evaluate(test_batches)
print("Loss = ",test_loss)
print("Accuracy = ",test_acc)

#%%
# Model saving
model.save('Image_segmentation_of_nuclei_images_model.h5')