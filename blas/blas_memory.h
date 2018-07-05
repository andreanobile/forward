#ifndef MEMORY_H
#define MEMORY_H

#ifdef __cplusplus
extern "C" {
#endif
void blas_lock(volatile unsigned long *address);
void blas_unlock(volatile unsigned long *address);
void *blas_memory_alloc(int procpos);
void blas_memory_free(void *free_area);

#ifdef __cplusplus
}
#endif
#endif // MEMORY_H

