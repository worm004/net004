#include <cmath>
#include "stdlib.h"
#include "LRNLayer.h"
LRNLayer::LRNLayer(){
}
LRNLayer::LRNLayer(const JsonValue& j):Layer(j){
	const JsonValue& attrs = j.jobj.at("attrs");
	alpha = attrs.jobj.at("alpha").jv.d;
	beta = attrs.jobj.at("beta").jv.d;
	local_size = attrs.jobj.at("local_size").jv.d;
}
void LRNLayer::show(){
	Layer::show();
	printf("  (alpha) %g\n",alpha);
	printf("  (beta) %g\n",beta);
	printf("  (local_size) %d\n",local_size);
}
void LRNLayer::setup_outputs(){
	outputs[0].set_shape(inputs[0]);
	inplace = false;
	setup_outputs_data();
	buffer.set_shape(1,outputs[0].c,outputs[0].h,outputs[0].w);
	buffer.alloc();
}
void LRNLayer::forward(){
	//show_inputs();
	const Blob& ib = inputs[0], &ob = outputs[0];
	float *buf_data = buffer.data, *idata = ib.data, *odata = ob.data;
	int chw = ib.chw(), hw = ib.hw(), halfn = local_size/2;
	memset(odata, 0, sizeof(float) * ob.nchw());
	for(int i=0;i<ib.n;++i){
		int index = chw * i;
		float *cur_idata = idata + index, *cur_odata = odata + index;
		for(int j=0;j<chw;++j)
			buf_data[j] = cur_idata[j] * cur_idata[j] * alpha / local_size;
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
	//show_outputs();
}
