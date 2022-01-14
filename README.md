<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">CNN Architectures</div>
<div align="center"><img src="https://github.com/Pradnya1208/CNN-Architectures/blob/main/output/overview.gif?raw=true"></div>


## Overview:
A Gold mine dataset for computer vision is the ImageNet dataset. It consists of about 14 M hand-labelled annotated images which contains over 22,000 day-to-day categories. Every year ImageNet competition is hosted in which the smaller version of this dataset (with 1000 categories) is used with an aim to accurately classify the images. Many winning solutions of the ImageNet Challenge have used state of the art convolutional neural network architectures to beat the best possible accuracy thresholds.


## Dataset:
[xception](https://www.kaggle.com/keras/xception)<br>
[VGG19](https://www.kaggle.com/keras/vgg19)<br>
[VGG16](https://www.kaggle.com/keras/vgg16)<br>
[ResNet50](https://www.kaggle.com/keras/resnet50)<br>
[inceptionv3](https://www.kaggle.com/keras/inceptionv3)<br>
[Fruits 360](https://www.kaggle.com/moltean/fruits)<br>
[Flowers Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition)<br>
## Implementation:

**Libraries:**  `NumPy` `pandas` `PIL` `seaborn` `keras`
### VGG16:
VGG16 was publised in 2014 and is one of the simplest (among the other cnn architectures used in Imagenet competition). It's Key Characteristics are:

- This network contains total 16 layers in which weights and bias parameters are learnt.
- A total of 13 convolutional layers are stacked one after the other and 3 dense layers for classification.
- The number of filters in the convolution layers follow an increasing pattern (similar to decoder architecture of autoencoder).
- The informative features are obtained by max pooling layers applied at different steps in the architecture.
- The dense layers comprises of 4096, 4096, and 1000 nodes each.
- The cons of this architecture are that it is slow to train and produces the model with very large size.

  The VGG16 architecture is given below:
  <img src="https://github.com/Pradnya1208/CNN-Architectures/blob/main/output/vgg16.PNG?raw=true">
  <br>
Keras library also provides the pre-trained model in which one can load the saved model weights, and use them for different purposes : transfer learning, image feature extraction, and object detection. We can load the model architecture given in the library, and then add all the weights to the respective layers.
```
from keras.applications.vgg16 import VGG16
vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
vgg16_model = VGG16(weights=vgg16_weights)
_get_predictions(vgg16_model)
```
<img src ="https://github.com/Pradnya1208/CNN-Architectures/blob/main/output/vgg16_.PNG?raw=true">
<br>

### VGG19:
VGG19 is a similar model architecure as VGG16 with three additional convolutional layers, it consists of a total of 16 Convolution layers and 3 dense layers. Following is the architecture of VGG19 model. In VGG networks, the use of 3 x 3 convolutions with stride 1 gives an effective receptive filed equivalent to 7 * 7. This means there are fewer parameters to train.
<br>
<img src="https://github.com/Pradnya1208/CNN-Architectures/blob/main/output/vgg19.PNG?raw=true">
<br>
```
from keras.applications.vgg19 import VGG19
vgg19_weights = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'
vgg19_model = VGG19(weights=vgg19_weights)
_get_predictions(vgg19_model,0)
```
<br>
<img src="https://github.com/Pradnya1208/CNN-Architectures/blob/main/output/vgg19_.PNG?raw=true">
<br>

### InceptionNet:
Also known as GoogleNet consists of total 22 layers and was the winning model of 2014 image net challenge.

- Inception modules are the fundamental block of InceptionNets. The key idea of inception module is to design good local network topology (network within a network)
- These modules or blocks acts as the multi-level feature extractor in which convolutions of different sizes are obtained to create a diversified feature map
- The inception modules also consists of 1 x 1 convolution blocks whose role is to perform dimentionaltiy reduction.
- By performing the 1x1 convolution, the inception block preserves the spatial dimentions but reduces the depth. So the overall network's dimentions are not increased exponentially.
- Apart from the regular output layer, this network also consists of two auxillary classification outputs which are used to inject gradients at lower layers.
<br>
The inception module is shown in the following figure:
<br>
<img src="https://github.com/Pradnya1208/CNN-Architectures/blob/main/output/inceptionNet.PNG?raw=true">
<br>


```
from keras.applications.inception_v3 import InceptionV3
inception_weights = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
inception_model = InceptionV3(weights=inception_weights)
_get_predictions(inception_model,1)
```

<img src="https://github.com/Pradnya1208/CNN-Architectures/blob/main/output/inceptionv3.PNG?raw=true">
<br>

### ResNet:
[ResNet](https://arxiv.org/pdf/1512.03385.pdf)<br>
```
from tensorflow.keras.applications.resnet50 import ResNet50

resnet_weights = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
resnet_model = ResNet50(weights='imagenet')
_get_predictions(resnet_model,0)
```
<br>
<img src="https://github.com/Pradnya1208/CNN-Architectures/blob/main/output/resnet.PNG?raw=true">
<br>

Checkout complete implementation [here](https://github.com/Pradnya1208/CNN-Architectures/blob/main/cnn-architectures.ipynb)


### Learnings:
`CNN architectures`
`Using pre-trained models`






## References:
[ResNet](https://arxiv.org/pdf/1512.03385.pdf)

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner




[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

