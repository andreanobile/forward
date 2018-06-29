/*********************************************************************/
/* Copyright 2018 Andrea Nobile                                      */
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
/*********************************************************************/


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
#include <immintrin.h>

#define FLOAT float
#define CNAME sgemm_itcopy

int CNAME(long m, long n, FLOAT *a, long lda, FLOAT *b){

    long i, j;

    FLOAT *aoffset;
    FLOAT *aoffset1, *aoffset2;
    FLOAT *boffset;

    FLOAT ctemp01, ctemp02, ctemp03, ctemp04;
    /*
    FLOAT ctemp05, ctemp06, ctemp07, ctemp08;
    FLOAT ctemp09, ctemp10, ctemp11, ctemp12;
    FLOAT ctemp13, ctemp14, ctemp15, ctemp16;
    FLOAT ctemp17, ctemp18, ctemp19, ctemp20;
    FLOAT ctemp21, ctemp22, ctemp23, ctemp24;
    FLOAT ctemp25, ctemp26, ctemp27, ctemp28;
    FLOAT ctemp29, ctemp30, ctemp31, ctemp32;
    */
    __m128 v0, v1, v2, v3;//, v4, v5, v6, v7;
    __m256 y0, y1, y2, y3;


    aoffset   = a;
    boffset   = b;

#if 0
    fprintf(stderr, "m = %d n = %d\n", m, n);
#endif

    j = (n >> 4);
    if (j > 0){
        do{
            aoffset1  = aoffset;
            aoffset2  = aoffset + lda;
            aoffset += 16;

            i = (m >> 1);
            if (i > 0){
                do{

                    y0 = _mm256_loadu_ps(aoffset1 +  0);
                    y1 = _mm256_loadu_ps(aoffset1 +  8);
                    y2 = _mm256_loadu_ps(aoffset2 +  0);
                    y3 = _mm256_loadu_ps(aoffset2 +  8);

                    _mm256_storeu_ps(boffset+0, y0);
                    _mm256_storeu_ps(boffset+8, y1);
                    _mm256_storeu_ps(boffset+16, y2);
                    _mm256_storeu_ps(boffset+24, y3);

                    /*
                    v0 = _mm_loadu_ps(aoffset1 +  0);
                    v1 = _mm_loadu_ps(aoffset1 +  4);
                    v2 = _mm_loadu_ps(aoffset1 +  8);
                    v3 = _mm_loadu_ps(aoffset1 +  12);





                    v4 = _mm_loadu_ps(aoffset2 +  0);
                    v5 = _mm_loadu_ps(aoffset2 +  4);
                    v6 = _mm_loadu_ps(aoffset2 +  8);
                    v7 = _mm_loadu_ps(aoffset2 +  12);



                    _mm_storeu_ps(boffset+0, v0);
                    _mm_storeu_ps(boffset+4, v1);
                    _mm_storeu_ps(boffset+8, v2);
                    _mm_storeu_ps(boffset+12, v3);

                    _mm_storeu_ps(boffset+16, v4);
                    _mm_storeu_ps(boffset+20, v5);
                    _mm_storeu_ps(boffset+24, v6);
                    _mm_storeu_ps(boffset+28, v7);

                    */


                    aoffset1 +=  2 * lda;
                    aoffset2 +=  2 * lda;
                    boffset   += 32;

                    i --;
                }while(i > 0);
            }

            if (m & 1){

                v0 = _mm_loadu_ps(aoffset1 +  0);
                v1 = _mm_loadu_ps(aoffset1 +  4);
                v2 = _mm_loadu_ps(aoffset1 +  8);
                v3 = _mm_loadu_ps(aoffset1 +  12);

                _mm_storeu_ps(boffset+0, v0);
                _mm_storeu_ps(boffset+4, v1);
                _mm_storeu_ps(boffset+8, v2);
                _mm_storeu_ps(boffset+12, v3);

                /*
                ctemp01 = *(aoffset1 +  0);
                ctemp02 = *(aoffset1 +  1);
                ctemp03 = *(aoffset1 +  2);
                ctemp04 = *(aoffset1 +  3);
                ctemp05 = *(aoffset1 +  4);
                ctemp06 = *(aoffset1 +  5);
                ctemp07 = *(aoffset1 +  6);
                ctemp08 = *(aoffset1 +  7);
                ctemp09 = *(aoffset1 +  8);
                ctemp10 = *(aoffset1 +  9);
                ctemp11 = *(aoffset1 + 10);
                ctemp12 = *(aoffset1 + 11);
                ctemp13 = *(aoffset1 + 12);
                ctemp14 = *(aoffset1 + 13);
                ctemp15 = *(aoffset1 + 14);
                ctemp16 = *(aoffset1 + 15);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp02;
                *(boffset +  2) = ctemp03;
                *(boffset +  3) = ctemp04;
                *(boffset +  4) = ctemp05;
                *(boffset +  5) = ctemp06;
                *(boffset +  6) = ctemp07;
                *(boffset +  7) = ctemp08;

                *(boffset +  8) = ctemp09;
                *(boffset +  9) = ctemp10;
                *(boffset + 10) = ctemp11;
                *(boffset + 11) = ctemp12;
                *(boffset + 12) = ctemp13;
                *(boffset + 13) = ctemp14;
                *(boffset + 14) = ctemp15;
                *(boffset + 15) = ctemp16;
                * */

                boffset   += 16;
            }

            j--;
        }while(j > 0);
    } /* end of if(j > 0) */

    if (n & 8){
        aoffset1  = aoffset;
        aoffset2  = aoffset + lda;
        aoffset += 8;

        i = (m >> 1);
        if (i > 0){
            do{

                v0 = _mm_loadu_ps(aoffset1 +  0);
                v1 = _mm_loadu_ps(aoffset1 +  4);
                v2 = _mm_loadu_ps(aoffset2 +  0);
                v3 = _mm_loadu_ps(aoffset2 +  4);

                _mm_storeu_ps(boffset+0, v0);
                _mm_storeu_ps(boffset+4, v1);
                _mm_storeu_ps(boffset+8, v2);
                _mm_storeu_ps(boffset+12, v3);

                /*
                ctemp01 = *(aoffset1 +  0);
                ctemp02 = *(aoffset1 +  1);
                ctemp03 = *(aoffset1 +  2);
                ctemp04 = *(aoffset1 +  3);
                ctemp05 = *(aoffset1 +  4);
                ctemp06 = *(aoffset1 +  5);
                ctemp07 = *(aoffset1 +  6);
                ctemp08 = *(aoffset1 +  7);

                ctemp09 = *(aoffset2 +  0);
                ctemp10 = *(aoffset2 +  1);
                ctemp11 = *(aoffset2 +  2);
                ctemp12 = *(aoffset2 +  3);
                ctemp13 = *(aoffset2 +  4);
                ctemp14 = *(aoffset2 +  5);
                ctemp15 = *(aoffset2 +  6);
                ctemp16 = *(aoffset2 +  7);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp02;
                *(boffset +  2) = ctemp03;
                *(boffset +  3) = ctemp04;
                *(boffset +  4) = ctemp05;
                *(boffset +  5) = ctemp06;
                *(boffset +  6) = ctemp07;
                *(boffset +  7) = ctemp08;

                *(boffset +  8) = ctemp09;
                *(boffset +  9) = ctemp10;
                *(boffset + 10) = ctemp11;
                *(boffset + 11) = ctemp12;
                *(boffset + 12) = ctemp13;
                *(boffset + 13) = ctemp14;
                *(boffset + 14) = ctemp15;
                *(boffset + 15) = ctemp16;
                */
                aoffset1 +=  2 * lda;
                aoffset2 +=  2 * lda;
                boffset   += 16;

                i --;
            }while(i > 0);
        }

        if (m & 1){

            v0 = _mm_loadu_ps(aoffset1 +  0);
            v1 = _mm_loadu_ps(aoffset1 +  4);

            _mm_storeu_ps(boffset+0, v0);
            _mm_storeu_ps(boffset+4, v1);

            /*
            ctemp01 = *(aoffset1 +  0);
            ctemp02 = *(aoffset1 +  1);
            ctemp03 = *(aoffset1 +  2);
            ctemp04 = *(aoffset1 +  3);
            ctemp05 = *(aoffset1 +  4);
            ctemp06 = *(aoffset1 +  5);
            ctemp07 = *(aoffset1 +  6);
            ctemp08 = *(aoffset1 +  7);

            *(boffset +  0) = ctemp01;
            *(boffset +  1) = ctemp02;
            *(boffset +  2) = ctemp03;
            *(boffset +  3) = ctemp04;
            *(boffset +  4) = ctemp05;
            *(boffset +  5) = ctemp06;
            *(boffset +  6) = ctemp07;
            *(boffset +  7) = ctemp08;
            */
            boffset   += 8;
        }
    }

    if (n & 4){
        aoffset1  = aoffset;
        aoffset2  = aoffset + lda;
        aoffset += 4;

        i = (m >> 1);
        if (i > 0){
            do{

                v0 = _mm_loadu_ps(aoffset1 +  0);
                v1 = _mm_loadu_ps(aoffset2 +  0);

                _mm_storeu_ps(boffset+0, v0);
                _mm_storeu_ps(boffset+4, v1);

                /*
                ctemp01 = *(aoffset1 +  0);
                ctemp02 = *(aoffset1 +  1);
                ctemp03 = *(aoffset1 +  2);
                ctemp04 = *(aoffset1 +  3);

                ctemp05 = *(aoffset2 +  0);
                ctemp06 = *(aoffset2 +  1);
                ctemp07 = *(aoffset2 +  2);
                ctemp08 = *(aoffset2 +  3);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp02;
                *(boffset +  2) = ctemp03;
                *(boffset +  3) = ctemp04;
                *(boffset +  4) = ctemp05;
                *(boffset +  5) = ctemp06;
                *(boffset +  6) = ctemp07;
                *(boffset +  7) = ctemp08;
                */
                aoffset1 +=  2 * lda;
                aoffset2 +=  2 * lda;
                boffset   += 8;

                i --;
            }while(i > 0);
        }

        if (m & 1){
            v0 = _mm_loadu_ps(aoffset1 +  0);
            _mm_storeu_ps(boffset+0, v0);
            /*
            ctemp01 = *(aoffset1 +  0);
            ctemp02 = *(aoffset1 +  1);
            ctemp03 = *(aoffset1 +  2);
            ctemp04 = *(aoffset1 +  3);

            *(boffset +  0) = ctemp01;
            *(boffset +  1) = ctemp02;
            *(boffset +  2) = ctemp03;
            *(boffset +  3) = ctemp04;
            */
            boffset   += 4;
        }
    }

    if (n & 2){
        aoffset1  = aoffset;
        aoffset2  = aoffset + lda;
        aoffset += 2;

        i = (m >> 1);
        if (i > 0){
            do{
                ctemp01 = *(aoffset1 +  0);
                ctemp02 = *(aoffset1 +  1);
                ctemp03 = *(aoffset2 +  0);
                ctemp04 = *(aoffset2 +  1);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp02;
                *(boffset +  2) = ctemp03;
                *(boffset +  3) = ctemp04;

                aoffset1 +=  2 * lda;
                aoffset2 +=  2 * lda;
                boffset   += 4;

                i --;
            }while(i > 0);
        }

        if (m & 1){
            ctemp01 = *(aoffset1 +  0);
            ctemp02 = *(aoffset1 +  1);

            *(boffset +  0) = ctemp01;
            *(boffset +  1) = ctemp02;
            boffset   += 2;
        }
    }

    if (n & 1){
        aoffset1  = aoffset;
        aoffset2  = aoffset + lda;

        i = (m >> 1);
        if (i > 0){
            do{
                ctemp01 = *(aoffset1 +  0);
                ctemp02 = *(aoffset2 +  0);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp02;

                aoffset1 +=  2 * lda;
                aoffset2 +=  2 * lda;
                boffset   += 2;

                i --;
            }while(i > 0);
        }

        if (m & 1){
            ctemp01 = *(aoffset1 +  0);
            *(boffset +  0) = ctemp01;
            boffset   += 1;
        }
    }

    return 0;
}
