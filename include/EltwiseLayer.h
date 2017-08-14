#ifndef ELTWISELAYER_H
#define ELTWISELAYER_H
#include "BaseLayer.h"
class EltwiseLayer: public Layer{
	public:
	EltwiseLayer(const std::string& name, const std::string&l0, const std::string& l1, const std::string& method,float f0, float f1);
	virtual ~EltwiseLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void setup_dif_shape();
	virtual void setup_dif_data();
	virtual void show() const;
	virtual int parameter_number();
	
	std::string method;
	float f0 = 1.0f, f1 = 1.0f;
};

#endif
