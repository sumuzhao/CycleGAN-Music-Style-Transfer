# Music Style Translation based on CycleGAN

- Built a CycleGAN-based model to realize music style transfer between different musical domains.
- Added extra discriminators to regularize generators to achieve clear style transfer and preserve original melody, which made our model learn more high-level features.
- Trained several genre classifiers separately, and combined them with subjective judgement to have more convincing evaluations.

For details, refer to our paper [Symbolic Music Genre Transfer with CycleGAN](IEEE 2018 30th International Conference on Tools with Artificial Intelligence (ICTAI)). 

For some examples in paper, we uploaded them to our YouTube channel. 
(https://www.youtube.com/channel/UCs-bI_NP7PrQaMV1AJ4A3HQ)

For the datasets, we uploaded them to our Google drive. 
(https://drive.google.com/drive/folders/1LS_R0cKxGue7NMg2Nh9bvem6SQPu5Fja)

### Model Architecture
Our model generally follows the same structures as CycleGAN, which consists of two GANs arranged in a cyclic fashion and trained in unison. 

<img src="imgs/Picture1.png" width="1000px"/>

G denotes generators, D denotes discriminators, A and B are two domains. Blue and red arrows denote domain transfers in the two opposite directions, and black arrows point to the loss functions. M denotes a dataset containing music from multiple domains, e.g., M is composed of A and B. x, x hat and x tilde respectively denote a real data sample from source domain, the same data sample after being transferred to target domain and the same data sample after being transferred back to the source domain. 

For the generator and the discriminator, their architectures are following:

<img src="imgs/Generator.png" width="800px"/>
<img src="imgs/Discriminator.png" width="400px"/>

For the style classifier, its architecture is following:

<img src="imgs/Classifier.png" width="400px"/>

All the network parameters can be found in our paper. 

## Datasets
In this project, we use music of three different styles which are Classic, Jazz and Pop. 
Originally, we collected a lot of songs of different genres from various sources. And after preprocessing, we got our final training datasets like this. Note that, To avoid introducing a bias due to the imbalance of genres, we reduce the amount of samples in the larger dataset to match that of the smaller one. 

<img src="imgs/Picture2.png" width="500px"/>

Here are some concrete preprocessing steps.

First, we convert the MIDI files to piano-rolls with two python packages pretty_midi and pypianoroll. We need to sample the MIDI files to discrete time and allow a matrix respresentation t * p. t denotes time steps and p denotes pitches. The sampling rate is 16 which means the smallest note is 16th note. We discard notes above C8 and below C0, getting 84 pitches. Considering the temporal structure, we use phrases consisting of 4 consecutive bars as training samples. So the matrix has shape 64 by 84. This why CNN is feasible for our task. 

Second, we want to retain as much of the content of the original songs as possible. Thus we merge all the tracks into one single piano track. We omit drum tracks because it will make sounds cluttered severely. And we don’t use symphonies which contains too many instruments. 

Third, we omit velocity information. We fixed every note on velocity 100, resulting in a binary-valued matrix. 1 for note on and 0 for note off. 

Last, we remove songs whose first beat does not start at time step 0 and time signature is not 4/4. We also filter songs which has time signature changes throughout the songs. 

## Training


## Update Results
The results of this implementation:

- Classic -> Jazz <br>
<img src="imgs/C2J.png" width="500px"/> 

- Jazz -> Classic <br>
<img src="imgs/J2C.png" width="500px"/> 
