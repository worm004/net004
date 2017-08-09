#include <cfloat>
#include "stdlib.h"
#include "LossLayer.h"
LossLayer::LossLayer(const std::string&name, 
			const std::string& method):
				method(method),
				Layer(name,"loss"){
}
LossLayer::~LossLayer(){
}
void LossLayer::backward_softmax(){
}
void LossLayer::forward_softmax(){
	int batch_size = predict.n, c = predict.c, hw = predict.hw();
	float *pdata = predict.data, *gdata = gt.data;
	float *maxdata = maxs.data, *sumdata = sums.data, *softmaxdata = softmaxblob.data;
	float *odata = outputs[0].data;
	//predict.show();
	//gt.show();
	float loss = 0.0f;
	for(int i=0;i<batch_size;++i){

		for(int k=0;k<hw;++k){
			maxdata[k] = pdata[k];
			for(int j=1;j<c;++j)
				maxdata[k] = std::max(maxdata[k],pdata[j*hw+k]);
		}

		for(int j=0;j<c;++j)
			for(int k=0;k<hw;++k)
				softmaxdata[j*hw+k] = exp(pdata[j*hw+k] - maxdata[k]);

		for(int k=0;k<hw;++k){
			sumdata[k] = softmaxdata[k];
			for(int j=1;j<c;++j)
				sumdata[k] += softmaxdata[j*hw + k];
		}

		for(int j=0;j<c;++j)
			for(int k=0;k<hw;++k)
				softmaxdata[j*hw +k] /= sumdata[k];

		for(int j=0;j<c;++j)
			for(int k=0;k<hw;++k)
				loss -= log(std::max(softmaxdata[int(gdata[k]) * hw + k], FLT_MIN));

		pdata += c*hw;
		gdata += gt.chw();
	}
	odata[0] = loss/predict.c;
	//printf("loss: %f\n",odata[0]);

}
void LossLayer::forward(){
	printf("forward: %s %s\n",type.c_str(), method.c_str());
	show_inputs();
	if(method == "softmax")
		forward_softmax();
	show_outputs();
	
}
void LossLayer::backward(){
	printf("backward: %s %s\n",type.c_str(), method.c_str());
	if(method == "softmax")
		backward_softmax();
}
void LossLayer::show()const {
	printf("[%s%s] name: %s\n",
			type.c_str(),("+"+method).c_str(), 
			name.c_str());
}
void LossLayer::setup_shape(){
	if( (inputs.size()!=2) || (input_difs.size()!=2)){
		printf("error: loss input blob number should be 2\n");
		exit(0);
	}
	else if(inputs[0].n != inputs[1].n){
		printf("error: two inputs should have same batch size in loss layer\n");
		exit(0);
	}
	if((inputs[0].type == "label") && (inputs[1].type != "label")){
		gt.set_shape(inputs[0]);
		gt.set_data(inputs[0].data);
		gt.type = inputs[0].type;
		predict.set_shape(inputs[1]);
		predict.set_data(inputs[1].data);
	}
	else if((inputs[0].type != "label") && (inputs[1].type == "label")){
		gt.set_shape(inputs[1]);
		gt.set_data(inputs[1].data);
		gt.type = inputs[1].type;
		predict.set_shape(inputs[0]);
		predict.set_data(inputs[0].data);
	}
	else{
		printf("error: loss layer needs one label\n");
		exit(0);
	}
	
	outputs.resize(1);
	output_difs.resize(1);
	outputs[0].set_shape(1, 1, 1, 1);
	output_difs[0].set_shape(outputs[0]);

	if(method == "softmax"){
		maxs.set_shape(1,1,predict.h,predict.w);
		sums.set_shape(1,1,predict.h,predict.w);
		softmaxblob.set_shape(1,predict.c,predict.h,predict.w);
	}

}
void LossLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: loss output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();
	output_difs[0].alloc();
	if(method == "softmax"){
		maxs.alloc();
		sums.alloc();
		softmaxblob.alloc();
	}
}
