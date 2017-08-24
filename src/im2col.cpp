#include <iostream>
#include "stdio.h"
#include "im2col.h"
void im2col(float * im, int c, int h, int w, float *des, int kernel, int stride, int pad){
	//printf("kernel: %d, stride: %d, pad: %d\n",kernel, stride, pad);
	//printf("h: %d, w: %d\n",h,w);
	int ii = 0;
	for(int iy = -pad; iy <= h + pad - kernel; iy += stride)
	for(int ix = -pad; ix <= w + pad - kernel; ix += stride)
	for(int oc = 0; oc < c; ++oc){
		for(int ik = 0; ik < kernel; ++ik){
			int y = iy + ik;
			if((y < 0) || (y>= h)){
				for(int jk = 0; jk < kernel; ++jk)
					des[ii++] = 0.0f;
			}
			else{
				for(int jk = 0; jk < kernel; ++jk){
					int x = ix + jk;
					if((x < 0)|| (x>=w)) des[ii++] = 0.0f;
					else des[ii++] = im[w * h * oc + y * w + x];
				}
			}
		}
	}
	
	//printf("%d = %d x %d x %d\n",ii,kernel * kernel, c, ii/kernel/kernel/c);
	//getchar();
}
void col2im(int c, int h, int w, float *im, float*des, int kernel_h,int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w){
	int ii = 0;
	for(int iy = -pad_h; iy <= h + pad_h - kernel_h; iy += stride_h)
	for(int ix = -pad_w; ix <= w + pad_w - kernel_w; ix += stride_w)
	for(int oc = 0; oc < c; ++oc)
	for(int ik = 0; ik < kernel_h; ++ik)
	for(int jk = 0; jk < kernel_w; ++jk,++ii){
		int y = iy + ik, x = ix + jk;
		if((x < 0)|| (x>=w) || (y<0) || (y>=h));
		else im[w*h*oc + y * w + x] += des[ii];
	}
}
void generate_table(int c, int h, int w, int *table, int kernel_h,int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w){
	int ii = 0;
	for(int iy = -pad_h; iy <= h + pad_h - kernel_h; iy += stride_h)
	for(int ix = -pad_w; ix <= w + pad_w - kernel_w; ix += stride_w)
	for(int oc = 0; oc < c; ++oc)
	for(int ik = 0; ik < kernel_h; ++ik)
	for(int jk = 0; jk < kernel_w; ++jk,++ii){
		int y = iy + ik, x = ix + jk;
		if((x < 0)|| (x>=w) || (y<0) || (y>=h)) table[ii] = -1;
		else table[ii] = w*h*oc + y * w + x;
	}
}
void generate_table_inv(int c, int h, int w, int *table, int kernel_h,int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w){
	int ii = 0;
	for(int iy = -pad_h; iy <= h + pad_h - kernel_h; iy += stride_h)
	for(int ix = -pad_w; ix <= w + pad_w - kernel_w; ix += stride_w)
	for(int oc = 0; oc < c; ++oc)
	for(int ik = 0; ik < kernel_h; ++ik)
	for(int jk = 0; jk < kernel_w; ++jk,++ii){
		int y = iy + ik, x = ix + jk;
		if((x < 0)|| (x>=w) || (y<0) || (y>=h)) continue;
		else table[w*h*oc + y * w + x] = ii;
	}
}
void im2col2(float * im, int *table, float* des, int count){
	for(int i=0;i<count;++i){
		int t = table[i];
		if(t >= 0) des[i] = im[t];
		else des[i] = 0.0f;
	}
}
