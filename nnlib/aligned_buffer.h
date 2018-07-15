#ifndef ALIGNED_BUFFER
#define ALIGNED_BUFFER

#include <stdlib.h>
#include <iostream>


template<typename T, int ALIGN>
struct AlignedBuffer
{
    AlignedBuffer(AlignedBuffer &b) = delete;
    AlignedBuffer(const AlignedBuffer &b) = delete;
    AlignedBuffer &operator=(AlignedBuffer &b) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &b) = delete;
    AlignedBuffer(AlignedBuffer &&b) = delete;
    AlignedBuffer &operator=(AlignedBuffer &&b) = delete;

    T *data;

    void clear()
    {
        if(data) {
            free(data);
        }
        data = nullptr;
    }

    void allocate(size_t nelem)
    {
        clear();

        int ret = posix_memalign((void**)&data, ALIGN, sizeof(T)*nelem);
        if(ret != 0) {
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

