#include <cmath>
#include "stdlib.h"
#include "CropLayer.h"
CropLayer::CropLayer(){}
CropLayer::CropLayer(const LayerUnit& u):Layer(u){
	float v;
	u.geta("axis",v); axis = v;
	u.geta("offset",v); offset = v;

	offsets.push_back(offset);
	for(int i=axis+1;i<4;++i) offsets.push_back(offset);
}
void CropLayer::show(){
	Layer::show();
	printf("  (axis) %d\n", axis);
	printf("  (offset) %d\n", offset);
}
void CropLayer::setup_outputs(){
	if(axis == 0) outputs[0].set_shape(inputs[1]);
	else if(axis == 1)outputs[0].set_shape(inputs[0].n,inputs[1].c,inputs[1].h,inputs[1].w);
	else if(axis == 2)outputs[0].set_shape(inputs[0].n,inputs[0].c,inputs[1].h,inputs[1].w);
	else if(axis == 3)outputs[0].set_shape(inputs[0].n,inputs[0].c,inputs[0].h,inputs[1].w);
	setup_outputs_data();
}
void CropLayer::forward(){
	int bs[4] = {0,0,0,0}, es[4] = {inputs[1].n,inputs[1].c,inputs[1].h,inputs[1].w};
	if(axis > 0) es[0] = inputs[0].n;
	if(axis > 1) es[1] = inputs[0].c;
	if(axis > 2) es[2] = inputs[0].h;

	for(int i=axis;i<4;++i){
		bs[i] = offsets[i-axis];
		es[i] += bs[i];
	}
	float * des = outputs[0].data, 
	      * src = inputs[0].data;	
	int ws[] = {inputs[0].chw(),inputs[0].hw(),inputs[0].w};
	for(int i0=bs[0], index = 0;i0<es[0];++i0)
	for(int i1=bs[1];i1<es[1];++i1)
	for(int i2=bs[2];i2<es[2];++i2)
	for(int i3=bs[3];i3<es[3];++i3,++index){
		des[index] = src[i3 + i2*ws[2] + i1*ws[1] + i0*ws[0]];
	}
}
