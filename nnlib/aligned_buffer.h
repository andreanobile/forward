#ifndef ALIGNED_BUFFER
#define ALIGNED_BUFFER

#include <stdlib.h>
#include <iostream>


template<typename T, int ALIGN>
struct AlignedBuffer
{
    T *data;

    void clear()
    {
        if(data) free(data);
    }


    void allocate(size_t nelem)
    {
        clear();

        int ret = posix_memalign((void**)&data, ALIGN, sizeof(T)*nelem);
        if(ret) {
            std::cout << "failed to allocate aligned buffer ! \n";
            abort();
        }
    }


    AlignedBuffer()
    {
        data = nullptr;
    }


    explicit AlignedBuffer(size_t nelem)
    {
        allocate(nelem);
    }


    ~AlignedBuffer()
    {
        clear();
    }


    float & operator[](size_t idx)
    {
        return data[idx];
    }
};


#endif // ALIGNED_BUFFER

