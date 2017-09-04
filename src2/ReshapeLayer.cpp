#include <cmath>
#include "stdlib.h"
#include "ReshapeLayer.h"
ReshapeLayer::ReshapeLayer(){}
ReshapeLayer::ReshapeLayer(const LayerUnit& u):Layer(u){
	float v;
	u.geta("d0",v); shape[0] = v;
	u.geta("d1",v); shape[1] = v;
	u.geta("d2",v); shape[2] = v;
	u.geta("d3",v); shape[3] = v;
}
void ReshapeLayer::show(){
	Layer::show();
	printf("  (shape) [%d %d %d %d]\n",shape[0],shape[1],shape[2],shape[3]);
}
void ReshapeLayer::setup_outputs(){
	if(shape[0] == 0) shape[0] = inputs[0].n;
	if(shape[1] == 0) shape[1] = inputs[0].c;
	if(shape[2] == 0) shape[2] = inputs[0].h;
	if(shape[3] == 0) shape[3] = inputs[0].w;
	while(1){
		int num = 0;
		for(int i=0;i<4;++i){
			if(shape[i] == -1)
				++num;
		}
		if(num == 1){
			for(int i=0;i<4;++i){
				if(shape[i] == -1){
					int total = 1;
					for(int j=0;j<4;++j){
						if(j == i) continue;
						total *= shape[j];
					}
					shape[i] = inputs[0].nchw()/total;
				}
			}
		}
		else if(num == 0)break;
		else{
			printf("err: cannot be two or more -1 in reshape\n");
			exit(0);
		}
	}
	outputs[0].set_shape(shape[0],shape[1],shape[2],shape[3]);
	setup_outputs_data();
}
void ReshapeLayer::forward(){
}
