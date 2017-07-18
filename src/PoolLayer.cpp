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
	//printf("forward: %s %s %s\n",type.c_str(), name.c_str(), method.c_str());
	//printf("stride: %d, kernel: %d, padding: %d\n",stride, kernel, padding);

	//printf("input:\n");
	//for(int k=0;k<1;++k){
	//	for(int i=0;i<2;++i){
	//		for(int j=0;j<inputs[0].w;++j)
	//			printf("%f ",inputs[0].data[inputs[0].h * inputs[0].w *k + i*inputs[0].w + j]);
	//		printf("\n");
	//	}
	//}

	float *input_data = inputs[0].data;
	float *output_data = outputs[0].data;
	int h = inputs[0].h, w = inputs[0].w;
	int ah = (h + padding * 2 - kernel)%kernel;
	int aw = (w + padding * 2 - kernel)%kernel;
	int ii=0;
	for(int k=0;k<inputs[0].c;++k)
	for(int i=-padding;i<=h+padding+ah-kernel;i+=stride)
	for(int j=-padding;j<=w+padding+aw-kernel;j+=stride){
		float val = 0.0;
		int n = 0;
		for(int ik = 0;ik<kernel;++ik)
		for(int jk = 0;jk<kernel;++jk){
			int y = i + ik, x = j + jk;
			if((y < 0) || (x < 0) || (y >= h) || (x >= w))
				continue;
			val += input_data[k*h*w + w*y + x];
			++n;
		}
		output_data[ii++] = val/n;
	}

	//printf("result:\n");
	//for(int k=0;k<outputs[0].c;++k){
	//	for(int i=0;i<outputs[0].h;++i){
	//		for(int j=0;j<outputs[0].w;++j)
	//			printf("%f ",outputs[0].data[outputs[0].h * outputs[0].w *k + i*outputs[0].w + j]);
	//		printf("\n");
	//		getchar();
	//	}
	//}
	getchar();

}
void PoolLayer::forward_maxpool(){
	//printf("stride: %d, kernel: %d, padding: %d\n",stride, kernel, padding);
	float *input_data = inputs[0].data;
	float *output_data = outputs[0].data;
	int h = inputs[0].h, w = inputs[0].w;
	int ah = (h + padding * 2 - kernel)%kernel;
	int aw = (w + padding * 2 - kernel)%kernel;
	int ii=0;
	for(int k=0;k<inputs[0].c;++k)
	for(int i=-padding;i<=h+padding+ah-kernel;i+=stride)
	for(int j=-padding;j<=w+padding+aw-kernel;j+=stride){
		float val = -1e10;
		for(int ik = 0;ik<kernel;++ik)
		for(int jk = 0;jk<kernel;++jk){
			int y = i + ik, x = j + jk;
			if((y < 0) || (x < 0) || (y >= h) || (x >= w))
				continue;
			val = std::max(val, input_data[k*h*w + w*y + x]);
			//printf("%f\n",input_data[k*h*w + w*y + x]);
			//getchar();
		}
		output_data[ii++] = val;
	}
}
void PoolLayer::forward(){
	printf("forward: %s %s %s\n",type.c_str(), name.c_str(), method.c_str());
	printf("input:\n");
	for(int k=0;k<1;++k){
		for(int i=0;i<2;++i){
			for(int j=0;j<inputs[0].w;++j)
				printf("%f ",inputs[0].data[inputs[0].h * inputs[0].w *k + i*inputs[0].w + j]);
		}
	}

	if(method == "max"){
		forward_maxpool();
	}
	if(method == "avg"){
		forward_avgpool();
	}

	
	printf("\noutput:\n");
	for(int k=0;k<1;++k){
		for(int i=0;i<2;++i){
			for(int j=0;j<outputs[0].w;++j)
				printf("%f ",outputs[0].data[outputs[0].h * outputs[0].w *k + i*outputs[0].w + j]);
		}
	}
	getchar();

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
