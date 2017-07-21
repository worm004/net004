#include "stdlib.h"
#include "ConvLayer.h"
#include "im2col.h"
#include "Accelerate/Accelerate.h"

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
	//printf("forward: %s %s\n",type.c_str(), name.c_str());
	Blob &input = inputs[0], 
	     &output = outputs[0];

	int batch_size = input.n,
	    istep = input.chw(), 
	    ncol = kernel * kernel * output.hw() * input.c,
	    w = kernel * kernel * input.c, 
	    nloc = output.hw();
	float * idata = input.data, 
	      * odata = output.data,
	      * col_data = col,
	      * weight_data = weight.data, 
	      * bias_data = bias.data;
	for(int b = 0;b < batch_size; ++b){
		im2col(idata, input.c, input.h, input.w, col_data, kernel, stride, padding);
		//im2col2(idata, table, col_data, kernel * kernel * outputs[0].hw() * inputs[0].c);

		cblas_sgemm(CblasRowMajor, 
				CblasNoTrans, CblasTrans, 
				filters, nloc, w,
				1.0f,
				weight_data, w,
				col_data, w,
				0.0,
				odata, nloc);

		for(int i = 0; i < filters; ++i)
		for(int j = 0; j < nloc; ++j)
			odata[i*nloc +j] += bias_data[i];

		idata += istep;
		odata += output.chw();
		col_data += ncol;
	}
	if(activity == "relu"){
		int nchw = output.nchw();
		float* odata = output.data;
		for(int i=0;i<nchw;++i) if(odata[i] < 0) odata[i] = 0;
	}

	//show_inputs();
	//show_outputs();
	//getchar();


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

	if(bias.nchw() != 0){
		printf("\tbias: ");
		bias.show();
	}
	if(weight.nchw() != 0){
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
	return weight.nchw() + bias.nchw();
}

void ConvLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: conv output blob number should be 1\n");
		exit(0);
	}

	// col
	int ncol = kernel * kernel * outputs[0].hw() * inputs[0].c * outputs[0].n;
	col = new float[ncol];
	memset(col, 0, sizeof(float) * ncol);

	//col_bias = new float[ncol];
	//memset(col_bia, 0, sizeof(float) * ncol);
	//ones = new float[outputs[0].];

	//table = new int[ncol];
	//memset(table, 0, sizeof(int) * kernel * kernel * outputs[0].hw() * inputs[0].c);
	//generate_table(inputs[0].c, inputs[0].h, inputs[0].w, table, kernel, stride, padding);

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
