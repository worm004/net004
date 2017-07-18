#include "stdio.h"
#include "im2col.h"
void im2col(float * im, int c, int h, int w, float *des, int kernel, int stride, int pad){
	//printf("kernel: %d, stride: %d, pad: %d\n",kernel, stride, pad);
	//printf("h: %d, w: %d\n",h,w);
	int ii = 0;
	for(int iy = -pad; iy <= h + pad - kernel; iy += stride)
	for(int ix = -pad; ix <= w + pad - kernel; ix += stride)
	for(int oc = 0; oc < c; ++oc)
	for(int ik = 0; ik < kernel; ++ik)
	for(int jk = 0; jk < kernel; ++jk){
		int y = iy + ik, x = ix + jk;
		if((y < 0) || (x < 0) || (y>=h) || (x>=w)) des[ii++] = 0.0f;
		else des[ii++] = im[w * h * oc + y * w + x];
	}
	
	//printf("%d = %d x %d x %d\n",ii,kernel * kernel, c, ii/kernel/kernel/c);
	//getchar();
}
