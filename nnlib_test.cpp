/********************************************************************************
 *  Copyright (C) <2018>  <Andrea Nobile>                                       *
 *                                                                              *
 *   This program is free software: you can redistribute it and/or modify       *
 *   it under the terms of the GNU Affero General Public License as             *
 *   published by the Free Software Foundation, either version 3 of the         *
 *   License, or (at your option) any later version.                            *
 *                                                                              *
 *  This program is distributed in the hope that it will be useful,             *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 *   GNU Affero General Public License for more details.                        *
 *                                                                              *
 *  You should have received a copy of the GNU Affero General Public License    *
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.      *
 *                                                                              *
 *******************************************************************************/

#include <cstdio>
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <math.h>
#include "nnlib_test_config.h"
#include "nnlib.h"
#include "caffe_loader.h"
#include "network_node.h"
#include "net.h"
#include "layer.h"
#include "layer_factory.h"
#include "load_jpeg.h"
#include "rescale_image.h"
#include "chrono.h"
#include "string_utils.h"

#include <gd.h>
#include <thread>

using namespace std;


vector<string> load_synset(const string &fname)
{
    ifstream f(fname.c_str());
    stringstream ss;
    vector<string> output;

    if (f.is_open()) {
        ss << f.rdbuf();
        f.close();
        while (!ss.eof()) {
            string line;
            getline(ss, line);

            auto p = line.find(" ", 0);
            if(p != line.npos) {
                output.push_back(line.substr(p+1, line.length()-p-1));
            }

        }
    }

    return output;
}


vector<string> load_places_categories(const string &fname)
{
    ifstream f(fname.c_str());
    stringstream ss;
    vector<string> output;

    if (f.is_open()) {
        ss << f.rdbuf();
        f.close();
        while (!ss.eof()) {
            string line;
            getline(ss, line);

            auto p = line.find("/", 1);
            if(p != line.npos) {
                output.push_back(split_string(line.substr(p+1, line.length()-p-1), " ")[0]);
            }

        }
    }

    return output;
}


void ndarray_cmp(ndarray *d0, ndarray *d1)
{
    float maxdiff = 0.0f;
    int maxi = -1;
    for(size_t i=0;i<d0->n_elements();i++) {
        float diff = fabsf(d0->get_data()[i] - d1->get_data()[i]);
        if(diff > maxdiff) {
            maxdiff = diff;
            maxi = i;
        }
    }
    if(maxi >=0 ) {
        cout << "max diff at " << maxi << " " << d0->get_data()[maxi] << " " << d1->get_data()[maxi] ;
        cout << " value = " << maxdiff << endl;
    }
}


struct PreprocessInfo
{
    size_t h;
    size_t w;
    int* perm_channels;
    float *means;
};


void load_and_preprocess_image(const string &fname, const PreprocessInfo &info, ndarray &input_image)
{
    JpegLoader image_loader;
    RGBImage image;

    image_loader.load_and_decode_jpeg(fname, image);

    int w = image.w;
    int h = image.h;
    size_t nh = info.h;
    size_t nw = info.w;

    vector<unsigned char> reordered_image(3*w*h);

    unsigned char * __restrict__ srcdata = (unsigned char*)reordered_image.data();
    unsigned char * __restrict__ data = image.data.data();

    for(int c=0;c<3;c++) {
        int offs = c*w*h;
        for(int j=0;j<h;j++) {
            for(int i=0;i<w;i++) {

                srcdata[offs+i] = data[(j*w + i)*3 + c];
            }
            offs+=w;
        }
    }

    vector<unsigned char> dv(3*nw*nh);
    unsigned char * __restrict__ dst = dv.data();

    int * __restrict__ perm_channels = info.perm_channels; // 2 1 0
    for(int c=0;c<3;c++) {
        int offss = perm_channels[c]*w*h;
        int offsd = c*nw*nh;
        rescale_image(srcdata+offss, w, h, dst+offsd, nw, nh, 'h');
    }

    float * __restrict__ means = info.means;
    float * __restrict__ idata = input_image.get_data();
    for(int c=0;c<3;c++) {
        size_t coffs = c*nw*nh;
        for(size_t im=0;im<nh*nw;im++) {
            idata[im+coffs] = ((float)dst[im+coffs]) - means[c];
        }
    }
}


void load_and_preprocess_image_gd(const string &fname,
                                  const PreprocessInfo &info,
                                  ndarray &input_image)
{
    FILE * __restrict__ fp = fopen(fname.c_str(), "rb");
    if(!fp) {
        cout << "error opening image file " << fname << endl;
        abort();
    }

    gdImagePtr in = gdImageCreateFromJpeg(fp);
    fclose(fp);

    gdImageSetInterpolationMethod(in, GD_BILINEAR_FIXED);
    gdImagePtr out = gdImageScale(in, info.w, info.h);

    int w = info.w;
    int h = info.h;

    if(!in->trueColor) {
        printf("image not truecolor! \n");
        exit(0);
    }

    float * __restrict__ idata = input_image.get_data();

    int pc0 = info.perm_channels[0];
    int pc1 = info.perm_channels[1];
    int pc2 = info.perm_channels[2];
    float m0 = info.means[0];
    float m1 = info.means[1];
    float m2 = info.means[2];

    int offs, pix;
    unsigned char cc[3];
    for(int j=0;j<h;j++) {
        for(int i=0;i<w;i++) {

            pix = gdImageGetPixel(out, i, j);
            cc[0] = gdTrueColorGetRed(pix);
            cc[1] = gdTrueColorGetGreen(pix);
            cc[2] = gdTrueColorGetBlue(pix);

            offs = 0*w*h + j*w;
            idata[offs+i] = ((float)cc[pc0]) - m0;
            offs = 1*w*h + j*w;
            idata[offs+i] = ((float)cc[pc1]) - m1;
            offs = 2*w*h + j*w;
            idata[offs+i] = ((float)cc[pc2]) - m2;

        }
    }

    gdImageDestroy(in);
    gdImageDestroy(out);
}


template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


void show_output(Net *net, vector<string> &classes_desc)
{
    auto out = net->get_output();
    float *out_data = out->get_data();

    size_t nel = out->n_elements();
    vector<float> vout(out_data, out_data+nel);

    vector<size_t> srt = sort_indexes(vout);

    size_t n_to_print = 5;
    for(size_t i=0;i<n_to_print;i++) {
        size_t idx = srt[nel-1-i];
        printf("%.2f %s \n", vout[idx], classes_desc[idx].c_str());
        //cout << vout[idx] << " " << desc[idx] << endl;
    }
    printf("\n");
}


void net_forward_on_image(Net *net, const string& imfname, const vector<size_t> &image_shape)
{
    ndarray input_image(image_shape);
    int pc[3] = {2, 1, 0};
    float means[3] = {103.0626238,  115.90288257, 123.15163084}; //= {103.0626238,  115.90288257, 123.15163084};
    PreprocessInfo info;
    info.h = image_shape[2];
    info.w = image_shape[3];
    info.perm_channels = pc;
    info.means = means;
    load_and_preprocess_image_gd(imfname, info, input_image);

    net->copy_to_input(&input_image);
    net->forward();
}


int main(int argc, char *argv[])
{
    string net_name;

    if(argc > 1) {
        net_name = string(argv[1]);
    } else {
        net_name = "r50";
    }


    string net_desc_filename;
    vector<string> classes;

    string networks_dir = "../networks";

    if(net_name == "places") {
        net_name = networks_dir + "/places_r152";
        net_desc_filename = net_name + "/" + "deploy_resnet152_places365.prototxt";
        classes = load_places_categories(net_name + "/" + "categories_places365.txt");
    } else {
        string nn = net_name.substr(1, net_name.length()-1);
        net_name = networks_dir + "/" + net_name;
        net_desc_filename= net_name + "/" + "ResNet-" + nn + "-deploy.prototxt";
        classes = load_synset(net_name + "/" + "synset_words.txt");
    }

    CaffeLoader loader;
    vector<size_t> image_shape({1, 3, 224, 224});
    vector<unique_ptr<Net>> vnet;

    vnet.push_back(loader.load_prototxt(net_desc_filename, net_name));
    vnet[0]->bind(image_shape);

    int nthreads = 4;
    for(int ithread=1;ithread<nthreads;ithread++)
    {
        vnet.emplace_back(new Net);
        vnet[ithread]->copy_net_sharing_weights(*vnet[0]);
        vnet[ithread]->bind(image_shape);
    }

    thread t[nthreads -1];

    if(argc > 2) {

        for (int kk=2;kk<argc;) {

            if((kk+nthreads-1)<argc && nthreads > 1) {

                for(int i=0;i<nthreads-1;i++) {
                    t[i] = thread(net_forward_on_image, vnet[i+1].get(), string(argv[kk+1+i]), image_shape);
                }

                net_forward_on_image(vnet[0].get(), string(argv[kk]), image_shape);

                for(int i=0;i<nthreads-1;i++) {
                    t[i].join();
                }

                for(int i=0;i<nthreads;i++) {
                    cout << kk-2+i  << " inference on image " << string(argv[kk]) << '\n';
                    show_output(vnet[i].get(), classes);
                }

                kk+=nthreads;

            } else {

                net_forward_on_image(vnet[0].get(), string(argv[kk]), image_shape);
                cout << kk-2  << " inference on image " << string(argv[kk]) << '\n';
                show_output(vnet[0].get(), classes);
                kk+=1;

            }
        }

    }

    return 0;
}
  
