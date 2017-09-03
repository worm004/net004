#include "FCLayer.h"
#include "Accelerate/Accelerate.h"
FCLayer::FCLayer(){}
FCLayer::FCLayer(const LayerUnit& u):Layer(u){
	float v;	
	u.geta("bias",v); bias = v;
	u.geta("num",v); num = v;
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
