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
		//im2col(idata, input.c, input.h, input.w, col_data, kernel, stride, padding);
		im2col2(idata, table, col_data, kernel * kernel * outputs[0].hw() * inputs[0].c);

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

		// do this in backward
		//for(int i=0;i<nchw;++i) 
		//	activity_mask[i] = 1;

		for(int i=0;i<nchw;++i) 
			if(odata[i] < 0.0f) {
				odata[i] = 0.0f;
			 	activity_mask[i] = 0;
			 }
	}

	//show_inputs();
	//show_outputs();
	//getchar();
}

void ConvLayer::backward(){
	printf("backward: %s %s\n",type.c_str(), name.c_str());
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
	int ncol = kernel * kernel * outputs[0].hw() * inputs[0].c * inputs[0].n;
	col = new float[ncol];
	memset(col, 0, sizeof(float) * ncol);

	table = new int[ncol];
	memset(table, 0, sizeof(int) * ncol / inputs[0].n);
	generate_table(inputs[0].c, inputs[0].h, inputs[0].w, table, kernel, stride, padding);

	// activity
	activity_mask = new bool[outputs[0].nchw()];
	memset(activity_mask, 0, sizeof(bool) * outputs[0].nchw());

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
