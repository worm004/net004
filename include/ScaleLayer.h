#ifndef SCALELAYER_H
#define SCALELAYER_H
#include "BaseLayer.h"
class ScaleLayer: public Layer{
	public:
	ScaleLayer(const std::string& name);
	virtual ~ScaleLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void show() const;
	virtual int parameter_number();

	Blob weight, bias, weight_dif, bias_dif;
};
#endif
