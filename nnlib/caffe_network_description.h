#ifndef CAFFENETWORKDESCRIPTION_H
#define CAFFENETWORKDESCRIPTION_H

#include <sstream>
#include "network_node.h"

class CaffeNetworkDescription
{
    std::map<char, char> open_of_close;
    std::stack<char> par_stack;

    void check_match(char ch);
    void init_open_of_close();
    void rewrite_network(const NetworkNode *node);

public:

    NetworkNode *root;

    CaffeNetworkDescription();
    CaffeNetworkDescription(std::stringstream &ss);
    ~CaffeNetworkDescription();

    void build_from_caffe_prototxt(std::stringstream &ss);
    void rewrite_network();


};

#endif // CAFFENETWORKDESCRIPTION_H
