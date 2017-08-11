#ifndef IM2COL_H
#define IM2COL_H
void im2col(float * im, int c, int h, int w, float *des, int kernel, int stride, int pad);
void im2col2(float * im, int *table, float* des, int count);
void generate_table(int c, int h, int w, int *table, int kernel_h,int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
#endif
