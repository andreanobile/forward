import caffe

#model_def = '../caffe_imagenet/ResNet-50-deploy.prototxt'
#model_data = '../caffe_imagenet/ResNet-50-model.caffemodel'

#model_def = 'ResNet-101-deploy.prototxt'
#model_data = 'ResNet-101-model.caffemodel'

model_def = 'ResNet-152-deploy.prototxt'
model_data = 'ResNet-152-model.caffemodel'

net = caffe.Net(model_def, model_data, caffe.TEST)

#net.params['fc1000'][0].data.tofile('fc_dump')

for key in net.params:
    for i in range(len(net.params[key])):
        s = 'r152/{}_{}.dat'.format(key, i)
        net.params[key][i].data.tofile(s)


