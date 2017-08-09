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
void generate_table(int c, int h, int w, int *table, int kernel, int stride, int pad){
	int ii = 0;
	for(int iy = -pad; iy <= h + pad - kernel; iy += stride)
	for(int ix = -pad; ix <= w + pad - kernel; ix += stride)
	for(int oc = 0; oc < c; ++oc)
	for(int ik = 0; ik < kernel; ++ik)
	for(int jk = 0; jk < kernel; ++jk,++ii){
		int y = iy + ik, x = ix + jk;
		if((x < 0)|| (x>=w) || (y<0) || (y>=h)) table[ii] = -1;
		else table[ii] = w*h*oc + y * w + x;
	}
}
void im2col2(float * im, int *table, float* des, int count){
	for(int i=0;i<count;++i){
		int t = table[i];
		if(t >= 0) des[i] = im[t];
		else des[i] = 0.0f;
	}

	//for(int i=0;i<7;++i){
	//for(int j=0;j<7;++j)
	//	printf(" %d",table[i*7+j]);
	//	printf("\n");
	//}
	//getchar();
}

