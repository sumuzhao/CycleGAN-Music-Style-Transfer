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
_𝐺_𝐴→𝐵_ and _𝐺_𝐵→𝐴_ are two generators which transfer data between two domains A and B. 
_𝐷_𝐴_ and _𝐷_𝐵_ are two discriminators which distinguish if data is real or fake. 
_𝐷_𝐴, 𝑚_ and _𝐷_𝐵, 𝑚_ are two extra discriminators which force generators to learn more high-level features. 

𝐴 and 𝐵 are two domains. Blue and red arrows denote domain transfers in the two opposite directions, and black arrows point to the loss functions. 
For blue arrows, _𝑥_𝐴_ denotes a real data sample from source domain 𝐴. __𝑥__𝐵_ denotes the same data sample after being transferred to target domain 𝐵. __𝑥__𝐴_ denotes the same data sample after being transferred back to the source domain 𝐴. The same for red arrows. 
𝑀 is a dataset containing music from multiple domains, e.g., 𝑀=𝐴∪𝐵. _𝑥_𝑀_ denotes a data sample from 𝑀. 


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
