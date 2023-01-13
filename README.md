# Unet Image Segmentation of Nuclei Images

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

### Summary
This project is an image segmentation project of datasets that contains nuclei images from datascience bowl 2018 competitions where masking is going to applied and later on predicted on the test dataset to get the accuracy of the model. This project serves as a practice to get a better understanding on image segmentation of objects in images and the different approaches that can be done on it. The goal is to get atleast an 80% prediction accuracy which was achieved as can be seen in the model performance below. The architecture of the model is developed by implementing transfer learning of pretrained keras model MobileNetV2 and Unet for a better image segmentation results.

### Model architecture
![Model architecture](https://user-images.githubusercontent.com/121662880/212262266-dfecbaff-8b66-4bd4-b46e-b28661ad28f8.png)

### Loss and accuracy value of the model
![Model loss and accuracy value](https://user-images.githubusercontent.com/121662880/212262412-80aad21b-6bd3-41ba-a2b9-1f4ef1abb840.PNG)

### Sample images of image segmentation prediction
![Mask prediction on images 1](https://user-images.githubusercontent.com/121662880/212262723-42990a0a-8cba-477c-8f46-bfbf76b09a1b.png)
![Mask prediction on images 2](https://user-images.githubusercontent.com/121662880/212262751-c6b2d07f-29f7-4cfb-ae23-f8e17a694d51.png)

### Tensorboard accuracy graph
![Tensorboard accuracy graph](https://user-images.githubusercontent.com/121662880/212262799-f1d29731-255c-451d-a5c7-90255600faee.PNG)

### Tensorboard loss fraph
![Tensorboard loss graph](https://user-images.githubusercontent.com/121662880/212262833-1d413788-7cad-4788-8f8c-5ffa5387ef38.PNG)

### Credits
The datasets was obtained from --> https://www.kaggle.com/competitions/data-science-bowl-2018/overview
