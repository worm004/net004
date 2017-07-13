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

	private:
	int stride = 0, padding = 0, kernel = 0;
	std::string method;
};
#endif
