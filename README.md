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
Our model looks like below. 
<img src="imgs/Picture1.png" width="1000px"/>

For the generator and the discriminator, their architectures are following:
<img src="imgs/Generator.png" width="800px"/>
<img src="imgs/Discriminator.png" width="400px"/>

For the style classifier, its architecture is following:

<img src="imgs/Classifier.png" width="400px"/>



## Update Results
The results of this implementation:

- Classic -> Jazz <br>
<img src="imgs/C2J.png" width="500px"/> 

- Jazz -> Classic <br>
<img src="imgs/J2C.png" width="500px"/> 
