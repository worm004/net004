#include "stdlib.h"
#include "LRNLayer.h"
#include "Accelerate/Accelerate.h"
LRNLayer::LRNLayer(
	const std::string&name, 
	int n, 
	float alpha, 
	float beta):
		n(n),
		beta(beta),
		alpha(alpha),
		Layer(name,"lrn"){
}
LRNLayer::~LRNLayer(){
}
void LRNLayer::forward(){
	//printf("forward: %s %s\n",type.c_str(), name.c_str());
	
	const Blob& ib = inputs[0], &ob = outputs[0];
	float *buf_data = buffer.data;
	float *idata = ib.data, *odata = ob.data;

	int chw = ib.chw(), hw = ib.hw(), halfn = n/2;
	memset(odata, 0, sizeof(float) * ob.nchw());

	for(int i=0;i<ib.n;++i){
		int index = chw * i;
		float *cur_idata = idata + index, *cur_odata = odata + index;

		for(int j=0;j<chw;++j)
			buf_data[j] = cur_idata[j] * cur_idata[j] * alpha / n;
		for(int k=0;(k<=halfn)&&(k<ib.c);++k)
			for(int j=0, bindex = k*hw;j<hw;++j) cur_odata[j] += buf_data[bindex + j];
		
		for(int c = 1;c<ib.c;++c){
			int oindex = c * hw, cb = -halfn + c -1, ce = halfn + c;
			for(int j=0;j<hw;++j)
				cur_odata[oindex + j] = cur_odata[oindex - hw + j];
			if(cb >= 0) 
				for(int j=0, bindex = cb*hw;j<hw;++j)
				cur_odata[oindex + j] -= buf_data[bindex + j];
			if(ce < ib.c) 
				for(int j=0, bindex = ce*hw;j<hw;++j)
					cur_odata[oindex + j] += buf_data[bindex + j];
		}
			
	}

	int nchw = ib.nchw();
	for(int i=0;i<nchw;++i)
		odata[i] = pow(1 + odata[i],-beta) * idata[i];
	//show_inputs();
	//show_outputs();
}
void LRNLayer::backward(){
}
void LRNLayer::show()const {
	printf("[%s] name: %s, n: %d, alpha: %.4f, beta: %.4f\n",
			type.c_str(), 
			name.c_str(),
			n,alpha,beta);
	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}
	if(outputs.size() == 1){
		printf("\toutput: ");
		outputs[0].show();
	}

}
void LRNLayer::setup_shape(){
	if(inputs.size()!=1){
		printf("error: lrn input blob number should be 1\n");
		exit(0);
	}

	// output
	const Blob& ib = inputs[0];
	outputs.resize(1);
	outputs[0].set_shape(ib);
	buffer.set_shape(1,outputs[0].c,outputs[0].h,outputs[0].w);

	if(is_train){
		if(input_difs.size()!=1){
			printf("error: lrn input blob number should be 1\n");
			exit(0);
		}
		output_difs.resize(1);
		output_difs[0].set_shape(outputs[0]);
	}
}
void LRNLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: lrn output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();

	buffer.alloc();

	if(is_train){
		if(output_difs.size()!=1){
			printf("error: lrn output blob number should be 1\n");
			exit(0);
		}
		output_difs[0].alloc();
	}
}
