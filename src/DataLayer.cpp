#include "DataLayer.h"
using namespace std;
DataLayer::DataLayer(
	const std::string&name,
	int n, 
	int c, 
	int h, 
	int w,
	const std::string& method): 
		n(n),
		c(c),
		h(h),
		w(w),
		method(method),
		Layer(name,"data"){
	outputs.resize(1);
	output_difs.resize(1);
	outputs[0].set_shape(n,c,h,w);
	outputs[0].alloc();
	outputs[0].type = method;
}
DataLayer::~DataLayer(){
}
void DataLayer::forward(){
}
void DataLayer::backward(){
}
void DataLayer::setup_shape(){
}
void DataLayer::setup_data(){
}
void DataLayer::show() const{
	const Blob& ob = outputs[0];
	printf("[%s] name: %s, n: %d, c: %d, h: %d, w: %d\n",
		type.c_str(), name.c_str(),
		ob.n,ob.c,ob.h,ob.w);
}
void DataLayer::add_image(unsigned char* data, int index, float mean_r, float mean_g, float mean_b){
	if(outputs.size() == 0) {
		printf("datalayer has not been setup (outputs.size() == 0)\n");
		return ;
	}
	int h = outputs[0].h, w = outputs[0].w, c = outputs[0].c;
	float *odata = outputs[0].data + index * w * h * c;
	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j){
		odata[(i*w+j) + w*h*0] = data[(i*w+j)*c+(c-0-1)] - mean_r;
		odata[(i*w+j) + w*h*1] = data[(i*w+j)*c+(c-1-1)] - mean_g;
		odata[(i*w+j) + w*h*2] = data[(i*w+j)*c+(c-2-1)] - mean_b;
	}
	//for(int k=0;k<c;++k)
	//	odata[(i*w+j) + w*h*k] = data[(i*w+j)*c+(c-k-1)] - 127;
	
}
void DataLayer::add_label(int label, int index){
	if(outputs.size() == 0) {
		printf("datalayer has not been setup (outputs.size() == 0)\n");
		return ;
	}
	float *ldata = outputs[0].data + index;
	ldata[0] = label;
}
