#include "stdlib.h"
#include "PoolLayer.h"
PoolLayer::PoolLayer(){}
PoolLayer::PoolLayer(const LayerUnit& u):Layer(u){
	float v;	
	u.geta("global",v); global = v;
	u.geta("kernel_size",v); kernel = v;
	u.geta("pad",v); pad = v;
	u.geta("stride",v); stride = v;
	u.geta("method",method);

	if(method == "max") f = &PoolLayer::forward_max;
	else if(method == "avg") f = &PoolLayer::forward_avg;
	else{
		printf("no method: %s in pool layer\n",method.c_str());
		exit(0);
	}
}
void PoolLayer::show(){
	Layer::show();
	printf("  (method) %s\n",method.c_str());
	if(global) printf("  (global) %d\n",int(global));
	if(!global){
		printf("  (kernel) %d\n",kernel);
		printf("  (pad) %d\n",pad);
		printf("  (stride) %d\n",stride);
	}
}
void PoolLayer::setup_outputs(){
	int ih = inputs[0].h, 
	    iw = inputs[0].w,
	    oh = i2o_ceil(ih,kernel,stride,pad),
	    ow = i2o_ceil(iw,kernel,stride,pad);
	outputs[0].set_shape(inputs[0].n, inputs[0].c, oh, ow);
	setup_outputs_data();
}
void PoolLayer::forward(){
	//show_inputs();
	(this->*f)();
	//show_outputs();
}
void PoolLayer::forward_max(){
	int h = inputs[0].h,
	    w = inputs[0].w,
	    hw = inputs[0].hw(),
	    bc = inputs[0].n * inputs[0].c,
	    bottom = ((h + pad * 2 - kernel+stride-1)/stride + 1)*stride - pad-1,
	    right = ((w + pad * 2 - kernel+stride-1)/stride + 1)*stride - pad-1;
	float *idata = inputs[0].data, *odata = outputs[0].data;
	for(int t = 0, cur = 0; t < bc; ++t, idata += hw)
	for(int i=-pad;i<=bottom;i+=stride)
	for(int j=-pad;j<=right;j+=stride){
		float val = -1e10;
		int index = -1;
		for(int ik = 0;ik<kernel;++ik){
			int y = i + ik;
			if((y < 0) || (y >= h)) continue;
			float *idatay = idata + w*y;
			for(int jk = 0;jk<kernel;++jk){
				int x = j + jk;
				if((x < 0) || (x >= w)) continue;
				float v = idatay[x];
				if(v > val){
					val = v;
					index = y*w + x;
				}
			}
		}
		odata[cur++] = val;
	}
}
void PoolLayer::forward_avg(){
	int h = inputs[0].h,
	    w = inputs[0].w,
	    bc = inputs[0].n * inputs[0].c,
	    bottom = ((h + pad * 2 - kernel+stride-1)/stride + 1)*stride - pad-1,
	    right = ((w + pad * 2 - kernel+stride-1)/stride + 1)*stride - pad-1;
	float *idata = inputs[0].data,
	      *odata = outputs[0].data;
	for(int t = 0, cur = 0; t < bc; ++t, idata += w*h)
	for(int i=-pad;i<=bottom;i+=stride)
	for(int j=-pad;j<=right;j+=stride){
		float val = 0.0f;
		int n=0;
		for(int ik = 0;ik<kernel;++ik)
		for(int jk = 0;jk<kernel;++jk){
			int y = i + ik, x = j + jk;
			if((y>=-pad) && (x >= -pad) && (y < h+pad) && (x < w + pad)) ++n;
			if((y < 0) || (x < 0) || (y >= h) || (x >= w))
				continue;
			val += idata[w*y + x];
		}
		if(n == 0) odata[cur++] = 0.0f;
		else odata[cur++] = val/n;
	}
}
