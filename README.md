# Reliable Retinal Edema Lesions Segmentation in OCT
# The code will be released when the paper is published.


Abstract: The accurate segmentation of retinal edema lesions from OCT images can greatly aid ophthalmologists in evaluating retinal diseases. However, due to the complicated pathological features of retinal edema lesions, resulting in severe regional scale imbalance between different symptoms, it remains a great challenge to accurately segment retinal edema lesions in OCT images. In addition, these deep learning-based segmentation methods are still black boxes for segmenting a lesion region, which makes the segmentation results may be unreliable. Aiming to solve these challenges, we propose a novel reliable multi-scale wavelet enhanced transformer network for retinal edema lesions segmentation in OCT images. In the proposed model, we first design a novel feature extractor network by integrating the newly proposed adaptive wavelet down-sampling module with the pre-trained ResNet blocks, which can generate a wavelet representation to avoid feature loss while enhancing the ability of the network to represent local multi-resolution detailed features. Meanwhile, we also develop a novel multi-scale transformer module to combine with the wavelet enhanced extractor network to further improve the model’s capacity of extracting the multi-scale global features of the retinal edema lesions. Besides, to further improve the model's robustness and make the segmentation results more reliable, we design a novel uncertainty estimation module appended to the final layer of the model to generate the segmentation results and provide the corresponding uncertainty estimation map. Finally, the proposed method is evaluated on the public database of AI-Challenge 2018 for retinal edema lesions segmentation, and the results indicate that the proposed method achieves better segmentation performance than other state-of-the-art networks. Particularly, the proposed method also can provide the corresponding uncertainty assessment for the segmentation results to make the segmentation results more reliable.
