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
		int group,
		const std::string& activity);
	virtual ~ConvLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void show() const;
	virtual int parameter_number();
	int get_kernel(){return kernel;}
	int get_filters(){return filters;}
	int get_padding(){return padding;}
	int get_stride(){return stride;}
	std::string& get_activity(){return activity;}

	Blob weight, weight_dif, bias, bias_dif;

	float* col = 0;
	int* table = 0;
	bool *activity_mask = 0;
	private:
	int kernel = 0, filters = 0, padding = 0, stride = 0, group = 1;
	std::string activity;
};

#endif
