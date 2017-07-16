#include "stdio.h"
#include "stdlib.h"
#include "Blob.h"
int Blob::total() const{
	return n * c * h * w;
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
	if(total() <= 0){
		printf("cannot alloc new blob: %d %d %d %d\n",n,c,h,w);
		exit(0);
	}
	if(data){
		delete [] data;
		data = 0;
	}
	data = new float[total()];
	owner = true;
}
void Blob::show() const{
	printf("n = %d, c = %d, h = %d, w = %d\n", n, c, h, w);
}
