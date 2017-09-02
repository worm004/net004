#include "stdlib.h"
#include "ConvLayer.h"
ConvLayer::ConvLayer(){}
ConvLayer::ConvLayer(const LayerUnit& u):Layer(u){
	float v;	
	u.geta("bias",v); bias = v;
	u.geta("group",v); group = v;
	u.geta("num",v); num = v;
	u.geta("kernel_size_h",v); kernel_size_h = v;
	u.geta("kernel_size_w",v); kernel_size_w = v;
	u.geta("pad_h",v); pad_h = v;
	u.geta("pad_w",v); pad_w = v;
	u.geta("stride_h",v); stride_h = v;
	u.geta("stride_w",v); stride_w = v;
	if(bias != (params.find("bias") != params.end())){
		printf("conv parameter wrong: bias\n");
		exit(0);
	}
}
void ConvLayer::show(){
	Layer::show();
	printf("  (num) %d\n",num);
	printf("  (kernel) %d %d\n",kernel_size_h,kernel_size_w);
	printf("  (pad) %d %d\n",pad_h,pad_w);
	printf("  (stride) %d %d\n",stride_h,stride_w);
	printf("  (group) %d\n",group);
	printf("  (bias) %d\n",int(bias));
}
void ConvLayer::setup_outputs(){
	inplace = false;
	int ih = inputs[0].h, 
	    iw = inputs[0].w,
	    oh = i2o_floor(ih,kernel_size_h,stride_h,pad_h),
	    ow = i2o_floor(iw,kernel_size_w,stride_w,pad_w);

	outputs[0].set_shape(inputs[0].n, num, oh, ow);
	outputs[0].alloc();
}
