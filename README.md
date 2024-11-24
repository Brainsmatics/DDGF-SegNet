# High-throughput mesoscopic optical imaging data processing and parsing using differential-guided filtered neural networks

We design a efficient deep differential guided filtering module (DDGF) by fusing multi-scale iterative differential guided filtering with deep learning, which effectively refines image details while mitigating background noise. Subsequently, by amalgamating DDGF with deep learning network, we propose a lightweight automatic segmentation method DDGF-SegNet, which demonstrates robust performance on our dataset, achieving Dice of 0.92, Precision of 0.98, Recall of 0.91, and Jaccard index of 0.86.

## Dataset

The datasets utilized in our experiments were derived from immunofuorescencestained C57BL/6 mouse brain images acquired through the array-fMOST, a represen-tative high-throughput mesoscale optical imaging technology. Specifically, the Array Brain 1 dataset comprises 24 mouse brains that were embedded and imaged collectively, the Array Brain 2 dataset contains 3 mouse brains imaged together, and theArray Brain 3 dataset includes 28 mouse brains that were imaged simultaneously. Thepixel spacing for these datasets is 0.65 x 0.65 x 3 um'. and the dimensions of a singlecoronal section image, once stitched, can reach up to 16,000 x 80,000 pixels?. Fig.4illustrates a typical example from this dataset, where the data were strip-imaged usingarray-fMOST, with corresponding binary labels manually delineated by experts.The arrayed mouse brain dataset comprises a total of 1,559 array-fMOST imageswith 720 images from Array Brain 1,720 from Array Brain 2, and 150 from ArrayBrain 3. wing to the variations in the dimensions of the stitched images from thethree datasets, we performed downsampling, cropping, and rotation adjustments onthe concatenated images. Subsequently, the processed data were logically partitionedfor subsequent training and testing procedures. The datasets used and analysed during the current study are available from the corresponding author on reasonablerequest.

## Implementation details

Since the entire automated high-throughput preprocessing pipeline is primarily divided into two stages—model segmentation and cropping box acquisition—the data flow moves from the segmentation model training and prediction to the binary images, which serve as the input for the cropping box acquisition algorithm. We tested the computational efficiency of the automated high-throughput preprocessing pipeline using a test set of 150 arrayed mouse brain images, prepared during the data preparation phase.Regarding the experimental hardware requirements, the deep learning segmentation model training and testing were conducted on a dedicated computer with the following configuration: Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz (12 cores), 256GB DDR4 (3200MHz) RAM, and NVIDIA Quadro RTX 6000 (24GB) GPU. The code was executed in Python 3.7, implemented based on their open-source repositories on GitHub.The testing of the proposed array data preprocessing pipeline was carried out on the high-performance computing platform at the Suzhou Brain Space Information Research Institute, Huazhong University of Science and Technology. The test utilized 16 computing nodes, each configured with: Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz (20 cores), 256GB DDR4 (3200MHz) RAM.

## Main references

1\. Yang Y, Jiang T, Jia X, et al. Whole-brain connectome of GABAergic neurons in the mouse zona incerta\[J\]. Neuroscience Bulletin, 2022, 38(11): 1315-1329.

2\. Muñoz-Castañeda R, Zingg B, Matho K S, et al. Cellular anatomy of the mouse primary motor cortex\[J\]. Nature, 2021, 598(7879): 159-166.

3\. Chen S, Liu G, Li A, et al. Three-dimensional mapping in multi-samples with large-scale imaging and multiplexed post staining\[J\]. Communications Biology, 2023, 6(1): 148.

4\. Li J, Jin P, Zhu J, et al. Multi-scale GCN-assisted two-stage network for joint segmentation of retinal layers and discs in peripapillary OCT images\[J\]. Biomedical Optics Express, 2021, 12(4): 2204-2220.

5\. S. Borkovkina, A. Camino, W. Janpongsri, M. V. Sarunic, and Y. Jian, “Real-time retinal layer segmentation of oct volumes with gpu accelerated inferencing using a compressed, low-latency neural network,” Biomed. Opt. Express 11(7), 3968 (2020).
