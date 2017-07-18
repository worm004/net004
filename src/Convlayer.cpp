#include "stdlib.h"
#include "ConvLayer.h"
#include "im2col.h"

ConvLayer::ConvLayer(
	const std::string&name, 
	int filters, 
	int kernel, 
	int stride, 
	int padding, 
	const std::string& activity):
		kernel(kernel), 
		filters(filters), 
		padding(padding), 
		stride(stride), 
		activity(activity), 
		Layer(name,"conv"){
}

ConvLayer::~ConvLayer(){
}

void ConvLayer::forward(){
	printf("forward: %s %s\n",type.c_str(), name.c_str());
	Blob &b = inputs[0];
	float * data = b.data;
	im2col(data, b.c, b.h, b.w, col, kernel, stride, padding);

	float *weight_data = weight.data;
	float *bias_data = bias.data;
	float *out_data = outputs[0].data;
	int w = kernel * kernel * b.c;
	int w1 = outputs[0].h * outputs[0].w;
	for(int i=0;i<filters;++i)
		for(int j=0;j<w1;++j){
			float val = 0.0f;
			for(int k=0;k<w;++k)
				val += weight_data[i * w + k] * col[j * w + k];
			out_data[i * w1 + j] = val + bias_data[i];
		}
	if(activity == "relu"){
		int total = outputs[0].total();
		for(int i=0;i<total;++i)
			if(outputs[0].data[i] < 0 )
				outputs[0].data[i] = 0;
	}

	//if(name == "conv1"){

	//	printf("filters: %d, kernel: %d, stride: %d, padding: %d, h: %d, w: %d\n", filters, kernel, stride, padding, outputs[0].h, outputs[0].w);
	//	printf("bias:\n");
	//	for(int i=0;i<bias.total();++i)
	//		printf("%f ",bias.data[i]);

	//	printf("\nweight:\n");
	//	for(int i=0;i<1;++i){
	//	for(int k=0;k<1/*b.c*/;++k)
	//	for(int j=0;j<kernel*kernel;++j)
	//		printf("%.2f ",weight.data[i*b.c*kernel*kernel + k*kernel*kernel+j]);
	//	}


		//printf("\ninput:\n");
		//for(int k=0;k<1;++k){
		//	for(int i=0;i<inputs[0].h;++i){
		//		for(int j=0;j<inputs[0].w;++j)
		//			printf("%f ",inputs[0].data[inputs[0].h * inputs[0].w *k + i*inputs[0].w + j]);
		//		printf("\n");
		//	}
		//	printf("\n");
		//}

	//	printf("\ncols:\n");
	//	int w = kernel*kernel, h = outputs[0].h * outputs[0].w;
	//	for(int i=0;i<1;++i){
	//		for(int k=0;k<1/*b.c*/;++k){
	//			for(int j=0;j<w;++j){
	//				printf("%.2f ",col[i * w * b.c + k * w + j]);
	//			}
	//			printf("\n");
	//		}
	//		//getchar();
	//	}

		//printf("\noutput:\n");
		//for(int k=0;k<1;++k){
		//	for(int i=0;i<1/*outputs[0].h*/;++i){
		//		for(int j=0;j<outputs[0].w;++j)
		//			printf("%f ",outputs[0].data[outputs[0].h * outputs[0].w *k + i*outputs[0].w + j]);
		//	}
		//}
		//getchar();
	//}


	//int w = kernel*kernel, h = outputs[0].h * outputs[0].w;
	//printf("col: kernel: %d, stride: %d, padding: %d, w: %d, h: %d\n", 
	//		kernel, stride, padding, w, h);

	
	//printf("result:\n");
	//for(int k=0;k<outputs[0].c;++k){
	//	for(int i=0;i<outputs[0].h;++i){
	//		for(int j=0;j<outputs[0].w;++j)
	//			printf("%f ",outputs[0].data[b.h * b.w *k + i*b.w + j]);
	//		//printf("\n");
	//	}
	//}
	//getchar();
}

void ConvLayer::backward(){
}

void ConvLayer::show()const {
	printf("[%s%s] name: %s, filters: %d, kernel: %d, stride: %d, padding: %d\n",
		type.c_str(),activity.empty()?"":("+"+activity).c_str(), 
		name.c_str(),
		filters,kernel,stride,padding);

	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}
	//printf("\tinput dif:");
	//input_difs[0].show();

	if(bias.total() != 0){
		printf("\tbias: ");
		bias.show();
	}
	if(weight.total() != 0){
		//printf("\tbias dif: ");
		//bias_dif.show();
		printf("\tweight: ");
		weight.show();
	}
	//printf("\tweight dif: ");
	//weight_dif.show();

	if(outputs.size() == 1){
		printf("\toutput: ");
		outputs[0].show();
	}
	//printf("\toutput dif:");
	//output_difs[0].show();
}

int ConvLayer::parameter_number(){
	return weight.total() + bias.total();
}

void ConvLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: conv output blob number should be 1\n");
		exit(0);
	}
	// col
	col = new float[kernel * kernel * outputs[0].h * outputs[0].w * inputs[0].c];
	memset(col, 0, sizeof(float) * kernel * kernel * outputs[0].h * outputs[0].w * inputs[0].c);
	
	// weight and bias
	weight.alloc();
	weight_dif.alloc();
	bias.alloc();
	bias_dif.alloc();

	// output
	outputs[0].alloc();
	output_difs[0].alloc();
}

void ConvLayer::setup_shape(){
	if( (inputs.size()!=1) || (input_difs.size()!=1)){
		printf("error: conv input blob number should be 1\n");
		exit(0);
	}
	
	// weight and bias
	const Blob& ib = inputs[0];
	weight.set_shape(1,ib.c * filters, kernel, kernel);
	weight_dif.set_shape(weight);
	bias.set_shape(1,filters,1,1);
	bias_dif.set_shape(bias);

	// output
	outputs.resize(1);
	output_difs.resize(1);
	int oh = Layer::i2o_floor(ib.h,kernel,stride,padding),
	    ow = Layer::i2o_floor(ib.w,kernel,stride,padding);
	outputs[0].set_shape(ib.n, filters, oh, ow);
	output_difs[0].set_shape(outputs[0]);
}
