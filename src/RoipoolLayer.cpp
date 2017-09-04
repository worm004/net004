#include <cfloat>
#include <cmath>
#include "stdlib.h"
#include "RoipoolLayer.h"
RoipoolLayer::RoipoolLayer(){}
RoipoolLayer::RoipoolLayer(const LayerUnit& u):Layer(u){
	float v;
	u.geta("h",v); h = v;
	u.geta("w",v); w = v;
	u.geta("s",s);
}
void RoipoolLayer::show(){
	Layer::show();
	printf("  (h) %d\n", h);
	printf("  (w) %d\n", w);
	printf("  (s) %g\n", s);
}
void RoipoolLayer::setup_outputs(){
	inplace = false;
	const Blob& ib0 = inputs[0];
	const Blob& ib1 = inputs[1];
	outputs.resize(1);
	outputs[0].set_shape(ib1.c/5*ib1.n, ib0.c, h, w);
	setup_outputs_data();
}
void RoipoolLayer::forward(){
	//show_inputs();
	int n = inputs[0].n, c = inputs[0].c, nroi = inputs[1].c/5, hw = inputs[0].hw();
	float *rdata = inputs[1].data, *ddata = inputs[0].data, *odata = outputs[0].data;
	int count = outputs[0].nchw();
	for(int i=0;i<count;++i) odata[i] = -FLT_MAX;
	for(int i=0;i<n;++i){
		for(int j=0;j<nroi;++j){
			int rindex = j*5,
				ry = int(rdata[rindex+2]*s+0.5f),
				rx = int(rdata[rindex+1]*s+0.5f);

			float bin_size_h = float(std::max(int(rdata[rindex+4]*s+0.5f) - ry + 1,1))/float(h),
			      bin_size_w = float(std::max(int(rdata[rindex+3]*s+0.5f) - rx + 1,1))/float(w);
			
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
