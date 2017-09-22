#include "stdlib.h"
#include "ConcatLayer.h"
ConcatLayer::ConcatLayer(){}
ConcatLayer::ConcatLayer(const JsonValue& j):Layer(j){
	const JsonValue& attrs = j.jobj.at("attrs");
	method = attrs.jobj.at("method").jv.s;
	if(method == "channel") f = &ConcatLayer::forward_channel;
	else{
		printf("no method: %s in concat layer\n",method.c_str());
		exit(0);
	}
}

void ConcatLayer::show(){
	Layer::show();
}
void ConcatLayer::setup_outputs(){
	const Blob& ib = inputs[0];
	int c = 0, h = inputs[0].h, w = inputs[0].w;
	for (const auto& i: inputs){
		c += i.c;
		if((i.h != h) || (i.w != w)){
			printf("error: concat inputs should have same w and h\n");
			exit(0);
		}
	}
	outputs[0].set_shape(ib.n, c, ib.h, ib.w);
	setup_outputs_data();
}
void ConcatLayer::forward_channel(){
	int n = inputs[0].n;
	Blob& ob = outputs[0];
	float *odata = ob.data;
	
	for(int i=0;i<n;++i){
		for(int j=0;j<inputs.size();++j){
			const Blob& ib = inputs[j];
			float *idata = ib.data + ib.chw() * i;
			memcpy(odata,idata,sizeof(float)*ib.chw());
			odata += ib.chw();
		}
	}
}
void ConcatLayer::forward(){
	(this->*f)();
}
