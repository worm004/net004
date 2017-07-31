#ifndef POOLLAYER_H
#define POOLLAYER_H
#include "BaseLayer.h"
class PoolLayer: public Layer{
	public:
	PoolLayer(
		const std::string&name, 
		int kernel, 
		int stride, 
		int padding, 
		const std::string& method);
	virtual ~PoolLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void show()const;
	void forward_maxpool();
	void forward_avgpool();
	void backward_maxpool();
	void backward_avgpool();
	int get_stride(){return stride;}
	int get_padding(){return padding;}
	int get_kernel(){return kernel;}
	std::string get_method(){return method;}

	private:
	int stride = 0, padding = 0, kernel = 0;
	std::string method;
	int * mask = 0;
};
#endif
