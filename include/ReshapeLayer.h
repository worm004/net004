#ifndef RESHAPELAYER_H
#define RESHAPELAYER_H
#include "BaseLayer.h"

class ReshapeLayer: public Layer{
	public:
	ReshapeLayer(const std::string& name, const std::vector<int>& p4);
	virtual ~ReshapeLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void setup_dif_shape();
	virtual void setup_dif_data();
	virtual void show() const;


	private:
	std::vector<int> shape;
};

#endif
