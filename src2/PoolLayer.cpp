#include "PoolLayer.h"
PoolLayer::PoolLayer(){}
PoolLayer::PoolLayer(const LayerUnit& u):Layer(u){
	float v;	
	u.geta("global",v); global = v;
	u.geta("kernel_size",v); kernel = v;
	u.geta("pad",v); pad = v;
	u.geta("stride",v); stride = v;
}
void PoolLayer::show(){
	Layer::show();
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
	if(inplace&&(ih == oh)&&(iw == ow))
		outputs[0].set_data(inputs[0].data);
	else{
		outputs[0].alloc();
		inplace = false;
	}
}
