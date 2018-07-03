# forward
a small and fast neural network inference engine

A package containing extracted networks to test this program is
available here: https://drive.google.com/open?id=1QWEXEsZO6OhqolNyzhf2AXxn_07jHnlt

The resnet network in places_r152 was taken from 
https://github.com/CSAILVision/places365 (their licence applies)

The resnet networks (r50 r101 r152) are trained on imagenet by MSRA  
https://github.com/KaimingHe/deep-residual-networks (their licence applies)


BUILD:
in the source directory use cmake.

$ cmake .

$ make


USAGE: 

download the extracted networks from the googledrive link. The test program looks into ../networks .
To run with the places 365 network on your pictures contained in the directory ~/Pictures :

$ ./nnlib_test places ~/Pictures/*.jpg

To run the resnet-50 network trained on imagenet:

$ ./nnlib_test r50  ~/Pictures/*.jpg

resnet-101

$ ./nnlib_test r101  ~/Pictures/*.jpg

resnet-152

$ ./nnlib_test r152  ~/Pictures/*.jpg



