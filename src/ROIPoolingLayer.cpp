#include <cfloat>
#include "stdlib.h"
#include "ROIPoolingLayer.h"
using namespace std;
ROIPoolingLayer::ROIPoolingLayer(
	const std::string&name, int h, int w, float scale, const std::vector<std::string>& names):
		h(h),
		w(w),
		scale(scale),
		Layer(name,"roipooling"){

	for(int i=0;i<names.size();++i)
		order[names[i]] = i;
	if(is_train) input_difs.resize(names.size());
	inputs.resize(names.size());
}
ROIPoolingLayer::~ROIPoolingLayer(){
}
void ROIPoolingLayer::forward(){
	//printf("forward: %s %s\n",type.c_str(), name.c_str());
	//show_inputs();
	
	int n = inputs[0].n, c = inputs[0].c, nroi = inputs[1].c/5, hw = inputs[0].hw();
	float *rdata = inputs[1].data, *ddata = inputs[0].data, *odata = outputs[0].data;

	//printf("input0: %d %d %d %d\n",n,c,inputs[0].h,inputs[0].w);
	//printf("input1: %d %d %d %d\n",inputs[1].n,inputs[1].c,inputs[1].h,inputs[1].w);
	//printf("scale: %g, h: %d, w: %d\n",scale,h,w);

	int count = outputs[0].nchw();
	for(int i=0;i<count;++i) odata[i] = -FLT_MAX;

	for(int i=0;i<n;++i){
		for(int j=0;j<nroi;++j){
			int rindex = j*5,
				ry = int(rdata[rindex+2]*scale+0.5f),
				rx = int(rdata[rindex+1]*scale+0.5f);

			float bin_size_h = float(std::max(int(rdata[rindex+4]*scale+0.5f) - ry + 1,1))/float(h),
			      bin_size_w = float(std::max(int(rdata[rindex+3]*scale+0.5f) - rx + 1,1))/float(w);
			
			for(int y = 0, index = 0;y<h;++y){
				for(int x=0;x<w;++x,++index){
					int hstart = std::min(std::max(int(y*bin_size_h)+ry,0),inputs[0].h),
					    wstart = std::min(std::max(int(x*bin_size_w)+rx,0),inputs[0].w),
					    hend =   std::min(std::max(int(ceil((y+1)*bin_size_h))+ry,0),inputs[0].h),
					    wend =   std::min(std::max(int(ceil((x+1)*bin_size_w))+rx,0),inputs[0].w);
					if((hstart >= hend) || (wstart >= wend)){
						for(int k=0;k<c;++k)
							odata[w*h*k+index] = 0;
					}
					else{
						for(int k=0;k<c;++k){
							float &v = odata[w*h*k+index];
							for(int yy=hstart;yy<hend;++yy)
							for(int xx=wstart;xx<wend;++xx){
								int loc = yy*inputs[0].w+xx + hw*k;
								if(v < ddata[loc]) v = ddata[loc];
							}
						}
					}
				}
			}
			odata += outputs[0].chw();
		}
		rdata += inputs[1].chw();
		ddata += inputs[0].chw();
	}
	//show_outputs();
}

void ROIPoolingLayer::backward(){
	printf("backward: %s %s\n",type.c_str(), name.c_str());
}
void ROIPoolingLayer::show()const {
	printf("[%s] name: %s, h: %d, w: %d, scale: %f\n",
			type.c_str(), 
			name.c_str(),
			h,w,scale);
	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}
	if(outputs.size() == 1){
		printf("\toutput: ");
		outputs[0].show();
	}
}
void ROIPoolingLayer::setup_shape(){
	if(inputs.size()!=2){
		printf("error: roipool input blob number should be 2\n");
		exit(0);
	}
	// output
	const Blob& ib0 = inputs[0];
	const Blob& ib1 = inputs[1];
	outputs.resize(1);
	outputs[0].set_shape(ib1.c/5*ib1.n, ib0.c, h, w);
}
void ROIPoolingLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: roipool output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();
}
void ROIPoolingLayer::setup_dif_shape(){
	if(input_difs.size()!=2){
		printf("error: roipool input blob number should be 2\n");
		exit(0);
	}
	output_difs.resize(1);
	output_difs[0].set_shape(outputs[0]);
}
void ROIPoolingLayer::setup_dif_data(){
	if(output_difs.size()!=1){
		printf("error: pool output blob number should be 1\n");
		exit(0);
	}
	output_difs[0].alloc();
}
