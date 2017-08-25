#include "stdlib.h"
#include "DConvLayer.h"
#include "im2col.h"
#include "Accelerate/Accelerate.h"

DConvLayer::DConvLayer(
	const std::string&name, 
	int filters, 
	int kernel, 
	int stride, 
	int padding, 
	int group,
	bool is_bias,
	const std::string& activity):
		Layer(name,"dconv"),
		kernel_h(kernel), 
		kernel_w(kernel), 
		filters(filters), 
		padding_h(padding), 
		padding_w(padding), 
		stride_h(stride), 
		stride_w(stride), 
		group(group),
		is_bias(is_bias),
		activity(activity) {
}
DConvLayer::DConvLayer(
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
		Layer(name,"dconv"),
		kernel_h(kernel_h), 
		kernel_w(kernel_w), 
		filters(filters), 
		padding_h(padding_h), 
		padding_w(padding_w), 
		stride_h(stride_h), 
		stride_w(stride_w), 
		group(group),
		is_bias(is_bias),
		activity(activity){
}

DConvLayer::~DConvLayer(){
}

void DConvLayer::forward(){
	//printf("forward: %s %s\n",type.c_str(), name.c_str());
	
	//show_inputs();
	
	Blob &input = inputs[0], 
	     &output = outputs[0];
	
	int batch_size = input.n,
	    istep = input.chw(), 
	    ncol = kernel_h * kernel_w * inputs[0].hw() * filters,
	    w = kernel_h * kernel_w * filters, 
	    nloc = input.hw();

	float * idata = input.data, 
	      * odata = output.data,
	      * col_data = col,
	      * weight_data = weight.data, 
	      * bias_data = bias.data;
	for(int b = 0;b < batch_size; ++b){
		cblas_sgemm(CblasRowMajor, 
				CblasTrans, CblasNoTrans, 
				nloc, w, input.c,
				1.0f,
				idata, nloc,
				weight_data, w,
				0.0,
				col_data, w);

		//printf("weight\n");
		//for(int i=0;i<w*input.c;++i)
		//	printf(" %g",weight_data[i]);
		//printf("\n");

		//printf("col\n");
		//for(int i=0;i<nloc;++i)
		//for(int j=0;j<w;++j)
		//	printf(" %g",col_data[i*w+j]);
		//printf("\n");

		col2im(outputs[0].c, outputs[0].h, outputs[0].w, odata,col_data, kernel_h,kernel_w, stride_h,stride_w, padding_h,padding_w);

		if(is_bias){
			for(int i = 0; i < filters; ++i)
			for(int j = 0; j < output.hw(); ++j)
				odata[i*output.hw() +j] += bias_data[i];
		}

		idata += istep;
		odata += output.chw();
		col_data += ncol;
	}

	//show_outputs();
}

void DConvLayer::backward(){
	printf("backward: %s %s\n",type.c_str(), name.c_str());
}

void DConvLayer::show()const {
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

int DConvLayer::parameter_number(){
	return weight.nchw() + bias.nchw();
}

void DConvLayer::setup_shape(){
	if(inputs.size()!=1){
		printf("error: dconv input blob number should be 1\n");
		exit(0);
	}
	const Blob& ib = inputs[0];
	weight.set_shape(filters,ib.c, kernel_h, kernel_w);
	if(is_bias) bias.set_shape(filters,1,1,1);
	outputs.resize(1);
	int oh = (ib.h-1)*stride_h-2*padding_h + kernel_h,
	    ow = (ib.w-1)*stride_w-2*padding_w + kernel_w;
	outputs[0].set_shape(ib.n, filters, oh, ow);
}
void DConvLayer::setup_dif_shape(){
	if(input_difs.size()!=1){
		printf("error: dconv input blob number should be 1\n");
		exit(0);
	}
	weight_dif.set_shape(weight);
	if(is_bias) bias_dif.set_shape(bias);
	output_difs.resize(1);
	output_difs[0].set_shape(outputs[0]);
}
void DConvLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: dconv output blob number should be 1\n");
		exit(0);
	}

	// col
	int ncol = kernel_h * kernel_w * inputs[0].hw() * filters * inputs[0].n;
	col = new float[ncol];

	// weight, bias and output
	weight.alloc();
	if(is_bias) bias.alloc();
	outputs[0].alloc();
	memset(outputs[0].data,0,sizeof(float)*outputs[0].nchw());
}
void DConvLayer::setup_dif_data(){
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

