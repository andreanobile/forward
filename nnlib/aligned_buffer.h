#ifndef ALIGNED_BUFFER
#define ALIGNED_BUFFER

#include <stdlib.h>
#include <iostream>

struct AlignedBuffer
{
    float *data;

    void clear()
    {
        if(data) free(data);
    }


    void allocate(size_t nelem)
    {
        clear();

        int ret = posix_memalign((void**)&data, 64, sizeof(float)*nelem);
        if(ret) {
            std::cout << "failed to allocte fmap_buffer ! \n";
            abort();
        }
    }


    AlignedBuffer()
    {
        data = nullptr;
    }


    AlignedBuffer(size_t nelem)
    {
        allocate(nelem);
    }


    ~AlignedBuffer()
    {
        clear();
    }

    float & operator[](size_t idx) const
    {
        return data[idx];
    }

    float & operator[](size_t idx)
    {
        return data[idx];
    }
};


#endif // ALIGNED_BUFFER

