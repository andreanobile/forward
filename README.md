# forward
a small and fast neural network inference engine

On my laptop I get over 1.6x speedup using this code (No MKL) compared to Caffe+MKL, both using 1 core.
This code scales almost lienearly with the number of cores by processing each image in a separate thread.
The amount of memory used is also very different, almost 4x less!

Caffe (resnet-50) used from python on the famous kitten picture:
Maximum resident set size (kbytes): 487384

forward on the same picture:
Maximum resident set size (kbytes): 136900


A package containing extracted networks to test this program is
available here: https://drive.google.com/open?id=1QWEXEsZO6OhqolNyzhf2AXxn_07jHnlt

The resnet network in places_r152 was taken from 
https://github.com/CSAILVision/places365 (their licence applies)

The resnet networks (r50 r101 r152) are trained on imagenet by MSRA  
https://github.com/KaimingHe/deep-residual-networks (their licence applies)


To run the program a HASWELL (supporting AVX, fma) or newer cpu is needed. A Linux system is also needed. 
The code depends on turbojpeg and libgd to load and rescale jpeg images.

BUILD:

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
$ cd ..
```

in the source directory use cmake.
USAGE: 

download the extracted networks from the googledrive link. The test program looks into ../networks .
To run with the places 365 network on your pictures contained in the directory ~/Pictures :
```
$ ./nnlib_test places ~/Pictures/*.jpg
```
To run the resnet-50 network trained on imagenet:
```
$ ./nnlib_test r50  ~/Pictures/*.jpg
```
resnet-101
```
$ ./nnlib_test r101  ~/Pictures/*.jpg
```
resnet-152
```
$ ./nnlib_test r152  ~/Pictures/*.jpg
```



