#ifndef CONVLAYER_H
#define CONVLAYER_H
#include "BaseLayer.h"

class ConvLayer: public Layer{
	public:
	ConvLayer(
		const std::string&name, 
		int filters, 
		int kernel, 
		int stride, 
		int padding,
		const std::string& activity);
	virtual ~ConvLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void show() const;
	virtual int parameter_number();

	Blob weight, weight_dif, bias, bias_dif;

	float* col = 0;
	float* col_bias = 0;
	float* ones = 0;

	int* table = 0;
	private:
	int kernel = 0, filters = 0, padding = 0, stride = 0;
	std::string activity;
};

#endif
