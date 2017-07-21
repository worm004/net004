#include "stdlib.h"
#include "PoolLayer.h"

PoolLayer::PoolLayer(
	const std::string&name, 
	int kernel, 
	int stride, 
	int padding, 
	const std::string& method):
		kernel(kernel),
		stride(stride),
		padding(padding),
		method(method),
		Layer(name,"pool"){
}
PoolLayer::~PoolLayer(){
}
void PoolLayer::forward_avgpool(){
	int h = inputs[0].h,
	    w = inputs[0].w,
	    bc = inputs[0].n * inputs[0].c,
	    bottom = (h + padding * 2 - kernel)%kernel + h + padding - kernel,
	    right = (w + padding * 2 - kernel)%kernel + w + padding - kernel;
	float *idata = inputs[0].data,
	      *odata = outputs[0].data;

	for(int t = 0, cur = 0; t < bc; ++t, idata += w*h)
	for(int i=-padding;i<=bottom;i+=stride)
	for(int j=-padding;j<=right;j+=stride){
		float val = 0.0f;
		int n = 0;
		for(int ik = 0;ik<kernel;++ik)
		for(int jk = 0;jk<kernel;++jk){
			int y = i + ik, x = j + jk;
			if((y < 0) || (x < 0) || (y >= h) || (x >= w))
				continue;
			val += idata[w*y + x];
			++n;
		}
		if(n == 0) odata[cur++] = 0.0f;
		else odata[cur++] = val/n;
	}
}
void PoolLayer::forward_maxpool(){
	int h = inputs[0].h,
	    w = inputs[0].w,
	    bc = inputs[0].n * inputs[0].c,
	    bottom = (h + padding * 2 - kernel)%kernel + h + padding - kernel,
	    right = (w + padding * 2 - kernel)%kernel + w + padding - kernel;
	float *idata = inputs[0].data,
	      *odata = outputs[0].data;

	for(int t = 0, cur = 0; t < bc; ++t, idata += w*h)
	for(int i=-padding;i<=bottom;i+=stride)
	for(int j=-padding;j<=right;j+=stride){
		float val = -1e10;
		for(int ik = 0;ik<kernel;++ik)
		for(int jk = 0;jk<kernel;++jk){
			int y = i + ik, x = j + jk;
			if((y < 0) || (x < 0) || (y >= h) || (x >= w))
				continue;
			val = std::max(val, idata[w*y + x]);
		}
		odata[cur++] = val;
	}
}
void PoolLayer::forward(){
	//printf("forward: %s %s %s\n",type.c_str(), name.c_str(), method.c_str());
	if(method == "max") forward_maxpool();
	else if(method == "avg") forward_avgpool();
	else printf("not implemented %s in activity layer\n",method.c_str());

	//show_inputs();
	//show_outputs();
	//getchar();
}
void PoolLayer::backward(){
}
void PoolLayer::show()const {
	printf("[%s%s] name: %s, kernel: %d, stride: %d, padding: %d\n",
			type.c_str(),("+"+method).c_str(), 
			name.c_str(),
			kernel,stride,padding);
	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}
	if(outputs.size() == 1){
		printf("\toutput: ");
		outputs[0].show();
	}
}
void PoolLayer::setup_shape(){
	if( (inputs.size()!=1) || (input_difs.size()!=1)){
		printf("error: pool input blob number should be 1\n");
		exit(0);
	}

	// output
	const Blob& ib = inputs[0];
	outputs.resize(1);
	output_difs.resize(1);
	int oh = Layer::i2o_ceil(ib.h,kernel,stride,padding),
	    ow = Layer::i2o_ceil(ib.w,kernel,stride,padding);
	outputs[0].set_shape(ib.n, ib.c, oh, ow);
	output_difs[0].set_shape(outputs[0]);
}
void PoolLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: pool output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();
	output_difs[0].alloc();
}
