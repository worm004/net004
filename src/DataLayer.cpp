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
void DataLayer::load_image(unsigned char* data){
	if(outputs.size() == 0) {
		printf("datalayer has not been setup (outputs.size() == 0)\n");
		return ;
	}
	int h = outputs[0].h, w = outputs[0].w, c = outputs[0].c;
	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j)
	for(int k=0;k<c;++k)
		outputs[0].data[(i*32+j) + 32*32*k] = data[(i*32+j)*c+(c-k-1)] - 127;
	
}
