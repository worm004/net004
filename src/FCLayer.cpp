#include "stdlib.h"
#include "FCLayer.h"
#include "Accelerate/Accelerate.h"

FCLayer::FCLayer(
	const std::string&name, 
	int n, 
	const std::string& activity):
		n(n), 
		activity(activity),
		Layer(name,"fc"){
}
FCLayer::~FCLayer(){
}
void FCLayer::forward(){
	//printf("forward: %s %s\n",type.c_str(), name.c_str());

	float * idata = inputs[0].data,
		* odata = outputs[0].data;
	int w = inputs[0].chw(), h = this->n,
	    batch_size = inputs[0].n;

	cblas_sgemm(CblasRowMajor, 
			CblasNoTrans, CblasTrans, 
			inputs[0].n, n, w,
			1.0f,
			idata, w,
			weight.data, w,
			0.0,
			odata, n);

	for(int b = 0; b < batch_size; ++b){
		float *weight_data = weight.data;
		for(int y = 0; y < h; ++y){
			odata[y] += bias.data[y];
		}
		odata += h;
		idata += w;
	}
	
	//show_inputs();
	//show_outputs();
	//getchar();
}
void FCLayer::backward(){
}
void FCLayer::show()const {
	printf("[%s%s] name: %s, n: %d\n",
			type.c_str(),activity.empty()?"":("+"+activity).c_str(), 
			name.c_str(),n);

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
void FCLayer::setup_shape(){
	if( (inputs.size()!=1) || (input_difs.size()!=1)){
		printf("error: fc input blob number should be 1\n");
		exit(0);
	}
	// weight and bias
	const Blob& ib = inputs[0];
	int in = ib.c*ib.h*ib.h;

	weight.set_shape(1,in * n , 1, 1);
	weight_dif.set_shape(weight);
	bias.set_shape(1,n,1,1);
	bias_dif.set_shape(bias);

	// output
	outputs.resize(1);
	output_difs.resize(1);
	outputs[0].set_shape(ib.n, n, 1, 1);
	output_difs[0].set_shape(outputs[0]);
}
void FCLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: fc output blob number should be 1\n");
		exit(0);
	}
	// weight and bias
	weight.alloc();
	weight_dif.alloc();
	bias.alloc();
	bias_dif.alloc();

	// output
	outputs[0].alloc();
	output_difs[0].alloc();
}
int FCLayer::parameter_number(){
	return weight.nchw() + bias.nchw();
}
