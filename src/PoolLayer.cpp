#include "stdlib.h"
#include "PoolLayer.h"
PoolLayer::PoolLayer(){}
PoolLayer::~PoolLayer(){
	if(bp_map){
		delete []bp_map;
		bp_map = 0;
	}
}
PoolLayer::PoolLayer(const JsonValue& j):Layer(j){
	const JsonValue& attrs = j.jobj.at("attrs");
	global  = attrs.jobj.at("global").jv.d;
	if(!global) kernel = attrs.jobj.at("kernel_size").jv.d;
	else kernel = -1;
	pad  = attrs.jobj.at("pad").jv.d;
	stride  = attrs.jobj.at("stride").jv.d;
	method  = attrs.jobj.at("method").jv.s;
	if(method == "max") {
		f = &PoolLayer::forward_max;
		bf = &PoolLayer::backward_max;
	}
	else if(method == "avg") {
		f = &PoolLayer::forward_avg;
		bf = &PoolLayer::backward_avg;
	}
	else{
		printf("no method: %s in pool layer\n",method.c_str());
		exit(0);
	}
}
void PoolLayer::show(){
	Layer::show();
	printf("  (method) %s\n",method.c_str());
	if(global) printf("  (global) %d\n",int(global));
	if(!global){
		printf("  (kernel) %d\n",kernel);
		printf("  (pad) %d\n",pad);
		printf("  (stride) %d\n",stride);
	}
}
void PoolLayer::setup_outputs(){
	if(kernel == -1) kernel = inputs[0].h;
	int ih = inputs[0].h, 
	    iw = inputs[0].w,
	    oh = i2o_ceil(ih,kernel,stride,pad),
	    ow = i2o_ceil(iw,kernel,stride,pad);
	outputs[0].set_shape(inputs[0].n, inputs[0].c, oh, ow);
	setup_outputs_data();
	bp_map = new int [outputs[0].nchw()];
}
void PoolLayer::forward(){
	//show_inputs();
	(this->*f)();
	//show_outputs();
}
void PoolLayer::backward(){
	//show_diff_outputs();
	(this->*bf)();
	//show_diff_inputs();
}
void PoolLayer::forward_max(){
	int h = inputs[0].h,
	    w = inputs[0].w,
	    hw = inputs[0].hw(),
	    bc = inputs[0].n * inputs[0].c,
	    bottom = ((h + pad * 2 - kernel+stride-1)/stride + 1)*stride - pad-1,
	    right = ((w + pad * 2 - kernel+stride-1)/stride + 1)*stride - pad-1;
	float *idata = inputs[0].data, *odata = outputs[0].data;
	for(int t = 0, cur = 0; t < bc; ++t, idata += hw)
	for(int i=-pad;i<=bottom;i+=stride)
	for(int j=-pad;j<=right;j+=stride){
		float val = -1e10;
		int index = -1;//bp only
		for(int ik = 0;ik<kernel;++ik){
			int y = i + ik;
			if((y < 0) || (y >= h)) continue;
			float *idatay = idata + w*y;
			for(int jk = 0;jk<kernel;++jk){
				int x = j + jk;
				if((x < 0) || (x >= w)) continue;
				float v = idatay[x];
				if(v > val){
					val = v;
					index = t*hw + y*w + x;// bp only
				}
			}
		}
		bp_map[cur] = index;//bp only
		odata[cur++] = val;
	}
}
void PoolLayer::forward_avg(){
	int h = inputs[0].h,
	    w = inputs[0].w,
	    bc = inputs[0].n * inputs[0].c,
	    bottom = ((h + pad * 2 - kernel+stride-1)/stride + 1)*stride - pad-1,
	    right = ((w + pad * 2 - kernel+stride-1)/stride + 1)*stride - pad-1;
	float *idata = inputs[0].data,
	      *odata = outputs[0].data;
	for(int t = 0, cur = 0; t < bc; ++t, idata += w*h)
	for(int i=-pad;i<=bottom;i+=stride)
	for(int j=-pad;j<=right;j+=stride){
		float val = 0.0f;
		int n=0;
		for(int ik = 0;ik<kernel;++ik)
		for(int jk = 0;jk<kernel;++jk){
			int y = i + ik, x = j + jk;
			if((y>=-pad) && (x >= -pad) && (y < h+pad) && (x < w + pad)) ++n;
			if((y < 0) || (x < 0) || (y >= h) || (x >= w))
				continue;
			val += idata[w*y + x];
		}
		bp_map[cur] = n;//bp only
		if(n == 0) odata[cur++] = 0.0f;
		else odata[cur++] = val/n;
	}
}
void PoolLayer::backward_max(){
	//printf("backward max\n");
	memset(diff_inputs[0].data,0,sizeof(float)*diff_inputs[0].nchw());
	int nchw = outputs[0].nchw();
	float *input_data = diff_inputs[0].data, *output_data = diff_outputs[0].data;
	memset(input_data,0,sizeof(float)*diff_inputs[0].nchw());
	for(int i=0;i<nchw;++i){
		if(bp_map[i]>=0) input_data[bp_map[i]] += output_data[i];
	}
}
void PoolLayer::backward_avg(){
	//printf("backward avg\n");
	int h = inputs[0].h,
	    w = inputs[0].w,
	    bc = inputs[0].n * inputs[0].c,
	    bottom = ((h + pad * 2 - kernel+stride-1)/stride + 1)*stride - pad-1,
	    right = ((w + pad * 2 - kernel+stride-1)/stride + 1)*stride - pad-1;
	float *idata = inputs[0].data,
	      *odata = outputs[0].data;
	float *input_data = diff_inputs[0].data, *output_data = diff_outputs[0].data;
	memset(input_data,0,sizeof(float)*diff_inputs[0].nchw());
	for(int t = 0, cur = 0; t < bc; ++t, input_data += w*h)
	for(int i=-pad;i<=bottom;i+=stride)
	for(int j=-pad;j<=right;j+=stride){
		int n=bp_map[cur];
		if(n == 0){
			++cur;
			continue;
		}
		float val = output_data[cur++]/n;
		for(int ik = 0;ik<kernel;++ik)
			for(int jk = 0;jk<kernel;++jk){
				int y = i + ik, x = j + jk;
				if((y < 0) || (x < 0) || (y >= h) || (x >= w))
					continue;
				input_data[w*y+x] += val;
			}
	}
}
