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
	int group,
	bool is_bias,
	const std::string& activity):
		kernel_h(kernel), 
		kernel_w(kernel), 
		filters(filters), 
		padding_h(padding), 
		padding_w(padding), 
		stride_h(stride), 
		stride_w(stride), 
		activity(activity), 
		group(group),
		is_bias(is_bias),
		Layer(name,"conv"){
}
ConvLayer::ConvLayer(
	const std::string&name, 
	int filters, 
	int kernel_h, 
	int kernel_w, 
	int stride_h, 
	int stride_w, 
	int padding_h,
	int padding_w,
	int group,
	bool is_bias,
	const std::string& activity):
		kernel_h(kernel_h), 
		kernel_w(kernel_w), 
		filters(filters), 
		padding_h(padding_h), 
		padding_w(padding_w), 
		stride_h(stride_h), 
		stride_w(stride_w), 
		activity(activity), 
		group(group),
		is_bias(is_bias),
		Layer(name,"conv"){
}

ConvLayer::~ConvLayer(){
}

void ConvLayer::forward(){
	printf("forward: %s %s\n",type.c_str(), name.c_str());
	Blob &input = inputs[0], 
	     &output = outputs[0];

	int batch_size = input.n,
	    istep = input.chw(), 
	    ncol = kernel_h * kernel_w * output.hw() * input.c,
	    w = kernel_h * kernel_w * input.c, 
	    nloc = output.hw();
	float * idata = input.data, 
	      * odata = output.data,
	      * col_data = col,
	      * weight_data = weight.data, 
	      * bias_data = bias.data;
	for(int b = 0;b < batch_size; ++b){
		//im2col(idata, input.c, input.h, input.w, col_data, kernel, stride, padding);
		//im2col2(idata, table, col_data, kernel * kernel * outputs[0].hw() * inputs[0].c);

		//cblas_sgemm(CblasRowMajor, 
		//		CblasNoTrans, CblasTrans, 
		//		filters, nloc, w,
		//		1.0f,
		//		weight_data, w,
		//		col_data, w,
		//		0.0,
		//		odata, nloc);
		
		for(int g=0;g<group;++g){
			im2col2(idata + input.chw()/group*g, table, col_data + ncol/group * g, kernel_h * kernel_w * outputs[0].hw() * inputs[0].c/group);

			cblas_sgemm(CblasRowMajor, 
					CblasNoTrans, CblasTrans, 
					filters/group, nloc, w/group,
					1.0f,
					weight_data + g * weight.nchw()/group, w/group,
					col_data + ncol/group * g, w/group,
					0.0,
					odata + output.chw()/group * g, nloc);

		}

		if(is_bias){
			//printf("bias\n");
			for(int i = 0; i < filters; ++i)
			for(int j = 0; j < nloc; ++j)
				odata[i*nloc +j] += bias_data[i];
		}

		idata += istep;
		odata += output.chw();
		col_data += ncol;
	}
	if(activity == "relu"){
		int nchw = output.nchw();
		float* odata = output.data;

		if(is_train){
		// do this in backward
		//for(int i=0;i<nchw;++i) 
		//	activity_mask[i] = 1;
			for(int i=0;i<nchw;++i) 
				if(odata[i] < 0.0f) {
					odata[i] = 0.0f;
					activity_mask[i] = 0;
				}
		}
		else{
			for(int i=0;i<nchw;++i) 
				if(odata[i] < 0.0f) odata[i] = 0.0f;
		}
	}
	if(name == "rpn_conv/3x3"){
		//show_inputs();
		//show_outputs();
	}
}

void ConvLayer::backward(){
	printf("backward: %s %s\n",type.c_str(), name.c_str());
}

void ConvLayer::show()const {
	printf("[%s%s%s] name: %s, filters: %d, kernel: %dx%d, stride: %dx%d, padding: %dx%d, group: %d\n",
		type.c_str(),activity.empty()?"":("+"+activity).c_str(),bias.data?"+bias":"",
		name.c_str(),
		filters,kernel_h,kernel_w,stride_h,stride_w,padding_h,padding_w,group);

	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}

	if(bias.nchw() != 0){
		printf("\tbias: ");
		bias.show();
	}
	if(weight.nchw() != 0){
		printf("\tweight: ");
		weight.show();
	}

	if(outputs.size() == 1){
		printf("\toutput: ");
		outputs[0].show();
	}
}

int ConvLayer::parameter_number(){
	return weight.nchw() + bias.nchw();
}

void ConvLayer::setup_shape(){
	if(inputs.size()!=1){
		printf("error: conv input blob number should be 1\n");
		exit(0);
	}
	const Blob& ib = inputs[0];
	weight.set_shape(filters,ib.c/group, kernel_h, kernel_w);
	if(is_bias) bias.set_shape(filters,1,1,1);
	outputs.resize(1);
	int oh = Layer::i2o_floor(ib.h,kernel_h,stride_h,padding_h),
	    ow = Layer::i2o_floor(ib.w,kernel_w,stride_w,padding_w);
	outputs[0].set_shape(ib.n, filters, oh, ow);
}
void ConvLayer::setup_dif_shape(){
	if(input_difs.size()!=1){
		printf("error: conv input blob number should be 1\n");
		exit(0);
	}
	weight_dif.set_shape(weight);
	if(is_bias) bias_dif.set_shape(bias);
	output_difs.resize(1);
	output_difs[0].set_shape(outputs[0]);
}
void ConvLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: conv output blob number should be 1\n");
		exit(0);
	}

	// col
	int ncol = kernel_h * kernel_w * outputs[0].hw() * inputs[0].c * inputs[0].n;
	col = new float[ncol];
	memset(col, 0, sizeof(float) * ncol);

	table = new int[ncol/inputs[0].n/group];
	memset(table, 0, sizeof(int) * ncol / inputs[0].n/group);
	generate_table(inputs[0].c/group, inputs[0].h, inputs[0].w, table, kernel_h,kernel_w, stride_h,stride_w, padding_h,padding_w);

	// weight, bias and output
	weight.alloc();
	if(is_bias) bias.alloc();
	outputs[0].alloc();

}
void ConvLayer::setup_dif_data(){
	if(output_difs.size()!=1){
		printf("error: conv output blob number should be 1\n");
		exit(0);
	}

	// activity
	activity_mask = new bool[outputs[0].nchw()];
	memset(activity_mask, 0, sizeof(bool) * outputs[0].nchw());

	weight_dif.alloc();
	if(is_bias) bias_dif.alloc();
	output_difs[0].alloc();
}
