#include "FCLayer.h"
#include "Accelerate/Accelerate.h"
FCLayer::FCLayer(){}
FCLayer::FCLayer(const JsonValue& j):Layer(j){
	const JsonValue& attrs = j.jobj.at("attrs");
	bias  = attrs.jobj.at("bias").jv.d;
	num  = attrs.jobj.at("num").jv.d;
}
void FCLayer::show(){
	Layer::show();
	printf("  (bias) %d\n",int(bias));
	printf("  (num) %d\n",int(num));
}

void FCLayer::setup_outputs(){
	outputs[0].set_shape(inputs[0].n, num, 1, 1);
	inplace = false;
	setup_outputs_data();
}
void FCLayer::forward(){
	//show_inputs();
	float * idata = inputs[0].data, * odata = outputs[0].data;
	int w = inputs[0].chw(), h = this->num, batch_size = inputs[0].n;
	Blob &bias_b = params["bias"], &weight_b = params["weight"];
	cblas_sgemm(CblasRowMajor, 
			CblasNoTrans, CblasTrans, 
			inputs[0].n, num, w,
			1.0f,
			idata, w,
			weight_b.data, w,
			0.0,
			odata, num);
	if(!bias) return;
	for(int b = 0; b < batch_size; ++b){
		float *weight_data = weight_b.data;
		for(int y = 0; y < h; ++y)
			odata[y] += bias_b.data[y];
		odata += h;
	}
	//show_outputs();
}
void FCLayer::backward(){
	//show_diff_outputs();
	if(bias){
		Blob& diff_bias = diff_params["bias"], &diff_output = diff_outputs[0];
		int batch_size = diff_output.n, n = diff_bias.n;
		float *output_data = diff_output.data, *bias_data = diff_bias.data;
		for(int i=0;i<n;++i){
			float sum = output_data[i];
			for(int b = 1; b<batch_size; ++b)
				sum += output_data[b*n+i];
			bias_data[i] = sum;
		}
	}
	int w = inputs[0].chw();
	Blob& diff_weight = diff_params["weight"], &diff_output = diff_outputs[0];
	cblas_sgemm(CblasRowMajor, 
			CblasTrans, CblasNoTrans, 
			num,w, diff_output.n,
			1.0f,
			diff_output.data, num,
			inputs[0].data, w,
			0.0,
			diff_weight.data, w);
	if(diff_inputs[0].data){
		Blob& weight = params["weight"];
		cblas_sgemm(CblasRowMajor, 
				CblasNoTrans, CblasNoTrans, 
				diff_output.n, w, num,
				1.0f,
				diff_output.data, num,
				weight.data, w,
				0.0,
				diff_inputs[0].data, w);
	}
	//show_diff_inputs();
	//show_diff_params();
}
