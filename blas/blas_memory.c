/*****************************************************************************
Copyright (c) 2018, Andrea Nobile
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************************/

/*****************************************************************************
Copyright (c) 2011-2014, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of
      its contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************************/

/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include <stdio.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <stdlib.h>
#include <sched.h>
#include "memory.h"


typedef unsigned long BLASULONG;
typedef long BLASLONG;

#define BUFFER_SIZE (32 << 20)
#define FIXED_PAGESIZE 4096

extern long int syscall (long int __sysno, ...);


static inline int my_mbind(void *addr, unsigned long len, int mode,
                           unsigned long *nodemask, unsigned long maxnode,
                           unsigned flags) {
#if defined (__LSB_VERSION__) || defined(ARCH_ZARCH)
    // So far,  LSB (Linux Standard Base) don't support syscall().
    // https://lsbbugs.linuxfoundation.org/show_bug.cgi?id=3482
    return 0;
#else

    //Fixed randomly SEGFAULT when nodemask==NULL with above Linux 2.6.34
    //      unsigned long null_nodemask=0;
    return syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);

#endif
}


struct release_t {
    void *address;
    void (*func)(struct release_t *);
    long attr;
};

//static int hugetlb_allocated = 0;

static struct release_t release_info[(4 * 2)];
static int release_pos = 0;
static BLASULONG alloc_lock = 0UL;

void blas_lock(volatile BLASULONG *address){

    int ret;

    do {
        while (*address) {sched_yield();};

        __asm__ __volatile__(
                    "xchgl %0, %1\n"
                    : "=r"(ret), "=m"(*address)
                    : "0"(1), "m"(*address)
                    : "memory");

    } while (ret);

}


void blas_unlock(volatile BLASULONG *address){

    *address = 0;
}


static void alloc_mmap_free(struct release_t *release){

    if (munmap(release -> address, BUFFER_SIZE)) {
        printf("OpenBLAS : munmap failed\n");
    }
}

#define MMAP_ACCESS (PROT_READ | PROT_WRITE)
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)

static void *alloc_mmap(void *address){

    void *map_address;

    if (address){
        map_address = mmap(address,
                           BUFFER_SIZE,
                           MMAP_ACCESS, MMAP_POLICY | MAP_FIXED, -1, 0);
    } else {
        map_address = mmap(address,
                           BUFFER_SIZE,
                           MMAP_ACCESS, MMAP_POLICY, -1, 0);
    }

    if (map_address != (void *)-1) {
        blas_lock(&alloc_lock);
        release_info[release_pos].address = map_address;
        release_info[release_pos].func = alloc_mmap_free;
        release_pos ++;
        blas_unlock(&alloc_lock);
    }

    my_mbind(map_address, BUFFER_SIZE, 1, NULL , 0, 0);

    return map_address;
}


static void alloc_malloc_free(struct release_t *release){

    free(release -> address);
}


static void *alloc_malloc(void *address){

    void *map_address;

    map_address = (void *)malloc(BUFFER_SIZE + FIXED_PAGESIZE);

    if (map_address == NULL) {
        map_address = (void *)-1;
    }

    if (map_address != (void *)-1) {
        release_info[release_pos].address = map_address;
        release_info[release_pos].func = alloc_malloc_free;
        release_pos ++;
    }

    return map_address;
}


static BLASULONG base_address = 0UL;


static volatile struct mem_t {
    BLASULONG lock;
    void *addr;
    int used;
    char dummy[40];
} memory[(4 * 2)];


static int memory_initialized = 0;


void *blas_memory_alloc_nolock(int unused) {
    void *map_address;
    map_address = (void *)malloc(BUFFER_SIZE + FIXED_PAGESIZE);
    return map_address;
}


void blas_memory_free_nolock(void * map_address) {
    free(map_address);
}


void *blas_memory_alloc(int procpos){

    int position = 0;
    void *map_address;
    void *(*memoryalloc[])(void *address) = {

            alloc_mmap,
            alloc_malloc,
            NULL};

    void *(**func)(void *address);

    blas_lock(&alloc_lock);

    if (!memory_initialized) {
        //blas_set_parameter(); // set sgemm_r:  sgemm_r = (((BUFFER_SIZE - ((SGEMM_P * SGEMM_Q *  4 + GEMM_OFFSET_A + GEMM_ALIGN) & ~GEMM_ALIGN)) / (SGEMM_Q *  4))- 15) & ~15;
        memory_initialized = 1;
    }

    blas_unlock(&alloc_lock);

    do {
        blas_lock(&memory[position].lock);
        if (!memory[position].used) goto allocation;

        blas_unlock(&memory[position].lock);
        position ++;

    } while (position < (4 * 2));

    goto error;

allocation :

    memory[position].used = 1;

    blas_unlock(&memory[position].lock);

    if (!memory[position].addr) {
        do {

            map_address = (void *)-1;

            func = &memoryalloc[1];

            while ((func != NULL) && (map_address == (void *) -1)) {
                map_address = (*func)((void *)base_address);
                func++;
            }


            if (((BLASLONG) map_address) == -1) base_address = 0UL;

            if (base_address) base_address += BUFFER_SIZE + FIXED_PAGESIZE;

        } while ((BLASLONG)map_address == -1);

        blas_lock(&alloc_lock);
        memory[position].addr = map_address;
        blas_unlock(&alloc_lock);

    }

    return (void *)memory[position].addr;

error:
    printf("BLAS : Program is Terminated. Because you tried to allocate too many memory regions.\n");

    return NULL;
}

void blas_memory_free(void *free_area){

    int position;

    position = 0;
    blas_lock(&alloc_lock);

    while ((position < (4 * 2)) && (memory[position].addr != free_area))
        position++;

    if (memory[position].addr != free_area) goto error;

    memory[position].used = 0;
    blas_unlock(&alloc_lock);

    return;

error:
    printf("BLAS : Bad memory unallocation! : %4d  %p\n", position, free_area);

    blas_unlock(&alloc_lock);

    return;
}


void blas_shutdown(void){

    int pos;

    blas_lock(&alloc_lock);

    for (pos = 0; pos < release_pos; pos ++) {
        release_info[pos].func(&release_info[pos]);
    }

    base_address = 0UL;

    for (pos = 0; pos < (4 * 2); pos ++){
        memory[pos].addr = (void *)0;
        memory[pos].used = 0;
        memory[pos].lock = 0;
    }

    blas_unlock(&alloc_lock);

    return;
}

static int blas_initialized = 0;


void __attribute__ ((constructor(101))) gotoblas_init(void) {

    if (blas_initialized) return;
    blas_initialized = 1;
}

void __attribute__ ((destructor(101))) gotoblas_quit(void) {

    if (blas_initialized == 0) return;

    blas_shutdown();
    blas_initialized = 0;
}
