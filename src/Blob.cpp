#include "stdio.h"
#include "stdlib.h"
#include "Blob.h"
int Blob::nchw() const{
	return n * c * h * w;
}
int Blob::chw() const{
	return c * h * w;
}
int Blob::hw() const{
	return h * w;
}
void Blob::set_shape(const Blob& b){
	set_shape(b.n, b.c, b.h, b.w);
}
void Blob::set_shape(int n, int c, int h, int w){
	this->n = n;
	this->c = c;
	this->h = h;
	this->w = w;
}
Blob::~Blob(){
	clear();
}
void Blob::clear(){
	if(data){
		if(owner) delete[] data;
		data = 0;
	}
}
void Blob::set_data(float * data){
	this->data = data;
	owner = false;
}
void Blob::alloc(){
	if(nchw() <= 0){
		printf("cannot alloc new blob: %d %d %d %d\n",n,c,h,w);
		exit(0);
	}
	if(data){
		delete [] data;
		data = 0;
	}
	data = new float[nchw()];
	owner = true;
}
void Blob::show() const{
	printf("n = %d, c = %d, h = %d, w = %d, type: %s\n", n, c, h, w, type.c_str());
}
void Blob::show_data(bool flat) const{
	show();
	if(!flat){
		int chw = this->chw();
		for(int b=0;b<n;++b){
			printf("[batch %d] ",b);
			for(int k=0;k<chw;++k)
				printf("%g ", data[b*chw + k]);
			printf("\n");
		}
	}else{
		int nchw = this->nchw();
		for(int i=0;i<nchw;++i)
			printf("%g ",data[i]);
		printf("\n");
	}
}
bool Blob::is_shape_same(const Blob&b){
	return (n == b.n) && (c == b.c) && (h == b.h) && (w == b.w);
}

