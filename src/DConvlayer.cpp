#include <cmath>
#include "stdlib.h"
#include "DConvLayer.h"
#include "im2col.h"
#include "Accelerate/Accelerate.h"

DConvLayer::DConvLayer(){}
DConvLayer::DConvLayer(const JsonValue& j):Layer(j){
	const JsonValue& attrs = j.jobj.at("attrs");
	bias  = attrs.jobj.at("bias").jv.d;
	group = attrs.jobj.at("group").jv.d;
	num   = attrs.jobj.at("num").jv.d;
	kernel_size_h = attrs.jobj.at("kernel_size_h").jv.d;
	kernel_size_w = attrs.jobj.at("kernel_size_w").jv.d;
	pad_h = attrs.jobj.at("pad_h").jv.d;
	pad_w = attrs.jobj.at("pad_w").jv.d;
	stride_h = attrs.jobj.at("stride_h").jv.d;
	stride_w = attrs.jobj.at("stride_w").jv.d;

	if(bias != ((j.jobj.find("params")!=j.jobj.end())&&(j.jobj.at("params").jobj.find("bias")!=j.jobj.at("params").jobj.end()))){
		printf("dconv parameter wrong: bias\n");
		exit(0);
	}
}
void DConvLayer::show(){
	Layer::show();
	printf("  (num) %d\n",num);
	printf("  (kernel) %d %d\n",kernel_size_h,kernel_size_w);
	printf("  (pad) %d %d\n",pad_h,pad_w);
	printf("  (stride) %d %d\n",stride_h,stride_w);
	printf("  (group) %d\n",group);
	printf("  (bias) %d\n",int(bias));
}
void DConvLayer::setup_outputs(){
	const Blob& ib = inputs[0];
	int oh = (ib.h-1)*stride_h-2*pad_h + kernel_size_h,
	    ow = (ib.w-1)*stride_w-2*pad_w + kernel_size_w;
	outputs[0].set_shape(ib.n, num, oh, ow);
	setup_outputs_data();
	int ncol = kernel_size_h * kernel_size_w * inputs[0].hw() * num * inputs[0].n;
	col = new float[ncol];
	memset(outputs[0].data,0,sizeof(float)*outputs[0].nchw());
}
void DConvLayer::forward(){
	Blob &input = inputs[0], 
	     &output = outputs[0];
	int batch_size = input.n,
	    istep = input.chw(), 
	    ncol = kernel_size_h * kernel_size_w * inputs[0].hw() * num,
	    w = kernel_size_h * kernel_size_w * num, 
	    nloc = input.hw();
	float * idata = input.data, 
	      * odata = output.data,
	      * col_data = col,
	      * weight_data = params["weight"].data, 
	      * bias_data = params["bias"].data;
	for(int b = 0;b < batch_size; ++b){
		cblas_sgemm(CblasRowMajor, 
				CblasTrans, CblasNoTrans, 
				nloc, w, input.c,
				1.0f,
				idata, nloc,
				weight_data, w,
				0.0,
				col_data, w);
		col2im(outputs[0].c, outputs[0].h, outputs[0].w, odata,col_data, kernel_size_h,kernel_size_w, stride_h,stride_w, pad_h,pad_w);
		if(bias){
			for(int i = 0; i < num; ++i)
			for(int j = 0; j < output.hw(); ++j)
				odata[i*output.hw() +j] += bias_data[i];
		}
		idata += istep;
		odata += output.chw();
		col_data += ncol;
	}
}
