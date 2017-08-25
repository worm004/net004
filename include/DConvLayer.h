#ifndef DCONVLAYER_H
#define DCONVLAYER_H
#include "BaseLayer.h"
class DConvLayer: public Layer{
	public:
	DConvLayer(
		const std::string&name, 
		int filters, 
		int kernel, 
		int stride, 
		int padding,
		int group,
		bool is_bias,
		const std::string& activity);
	DConvLayer(
		const std::string&name, 
		int filters, 
		int kernel_h, 
		int kernel_w, 
		int stride_h, 
		int stride_w, 
		int padding_h,
		int padding_w,
		int group,
		bool is_bias,
		const std::string& activity);
	virtual ~DConvLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void setup_dif_shape();
	virtual void setup_dif_data();
	virtual void show() const;
	virtual int parameter_number();
	int get_kernel_h(){return kernel_h;}
	int get_kernel_w(){return kernel_w;}
	int get_filters(){return filters;}
	int get_padding_h(){return padding_h;}
	int get_padding_w(){return padding_w;}
	int get_stride_h(){return stride_h;}
	int get_stride_w(){return stride_w;}
	std::string& get_activity(){return activity;}

	Blob weight, weight_dif, bias, bias_dif;

	float* col = 0;
	bool *activity_mask = 0;
	private:
	int kernel_h = 0, kernel_w = 0, 
		filters = 0, 
		padding_h = 0, padding_w = 0,
		stride_h = 0, stride_w = 0, 
		group = 1;

	bool is_bias = 0;
	std::string activity;
};
#endif
