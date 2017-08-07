#ifndef SPLITLAYER_H
#define SPLITLAYER_H
#include "BaseLayer.h"
class SplitLayer: public Layer{
	public:
	SplitLayer( const std::string&name);

	virtual ~SplitLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void show() const;
	virtual int parameter_number();
};

#endif
