project(assembler C ASM)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mfma")
add_library(nnlib caffe_network_description.cpp load_jpeg.cpp ndarray.cpp chrono.cpp net.cpp layer_factory.cpp file_utils.cpp string_utils.cpp rename_table.cpp caffe_loader.cpp conv_layer.cpp scale_layer.cpp batchnorm_layer.cpp layer.cpp relu_layer.cpp input_layer.cpp pooling_layer.cpp fc_layer.cpp softmax_layer.cpp eltwise_layer.cpp matvec.cpp dot.cpp filter_rcg.c)

