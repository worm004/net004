#include "stdlib.h"
#include "CropLayer.h"
using namespace std;
CropLayer::CropLayer(const std::string&name, int axis, const std::vector<int>& offset, const std::vector<std::string>& names):Layer(name,"crop"), axis(axis){
	for(int i=0;i<offset.size();++i)
		this->offset.push_back(offset[i]);
	if(axis > 3){
		printf("axis should be <= 3 in croplayer\n");
		exit(0);
	}
	if(offset.size()==1) {
		for(int i=axis+1;i<4;++i) 
			this->offset.push_back(offset[0]);
	}
	if(this->offset.size() != 4-axis){
		printf("croplayer input param error: axis: %d, len(offset): %d\n",axis, (int)this->offset.size());
		exit(0);
	}

	for(int i=0;i<names.size();++i)
		order[names[i]] = i;
	if(is_train) input_difs.resize(names.size());
	inputs.resize(names.size());
}
CropLayer::~CropLayer(){
}
void CropLayer::forward(){
	//printf("forward: %s %s\n",type.c_str(), name.c_str());
	//show_inputs();
	int bs[4] = {0,0,0,0}, es[4] = {inputs[1].n,inputs[1].c,inputs[1].h,inputs[1].w};
	if(axis > 0) es[0] = inputs[0].n;
	if(axis > 1) es[1] = inputs[0].c;
	if(axis > 2) es[2] = inputs[0].h;

	for(int i=0;i<4;++i){
		if(i<axis) continue;
		bs[i] = offset[i-axis];
		es[i] += bs[i];
		//printf("%d %d\n",bs[i],es[i]);
	}

	float* des = outputs[0].data, * src = inputs[0].data;	
	int ws[] = {inputs[0].chw(),inputs[0].hw(),inputs[0].w};
	for(int i0=bs[0], index = 0;i0<es[0];++i0)
	for(int i1=bs[1];i1<es[1];++i1)
	for(int i2=bs[2];i2<es[2];++i2)
	for(int i3=bs[3];i3<es[3];++i3,++index){
		des[index] = src[i3 + i2*ws[2] + i1*ws[1] + i0*ws[0]];
	}

	//show_outputs();
}
void CropLayer::backward(){
	printf("backward: %s %s\n",type.c_str(), name.c_str());
}
void CropLayer::setup_shape(){
	if(inputs.size()!=2){
		printf("error: crop input blob number should be 2\n");
		exit(0);
	}
	// output
	outputs.resize(1);
	if(axis == 0) outputs[0].set_shape(inputs[1]);
	else if(axis == 1)outputs[0].set_shape(inputs[0].n,inputs[1].c,inputs[1].h,inputs[1].w);
	else if(axis == 2)outputs[0].set_shape(inputs[0].n,inputs[0].c,inputs[1].h,inputs[1].w);
	else if(axis == 3)outputs[0].set_shape(inputs[0].n,inputs[0].c,inputs[0].h,inputs[1].w);
}
void CropLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: crop output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();

}
void CropLayer::setup_dif_shape(){
	if(input_difs.size()!=2){
		printf("error: crop input blob number should be 2\n");
		exit(0);
	}
	output_difs.resize(1);
	output_difs[0].set_shape(outputs[0]);
}
void CropLayer::setup_dif_data(){
	if(output_difs.size()!=1){
		printf("error: crop output blob number should be 1\n");
		exit(0);
	}
	output_difs[0].alloc();
}

void CropLayer::show()const {
	printf("[%s] name: %s\n", type.c_str(), name.c_str());

	if(inputs.size()){
		printf("\tinput: \n");
		for(int i=0;i<inputs.size();++i){
			printf("\t");
			inputs[i].show();
		}
	}

	if(outputs.size()){
		printf("\toutput: \n");
		for(int i=0;i<outputs.size();++i){
			printf("\t");
			outputs[i].show();
		}
	}
}

