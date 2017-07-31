#ifndef LOSSLAYER_H
#define LOSSLAYER_H
#include "BaseLayer.h"
class LossLayer: public Layer{
	public:
	LossLayer(const std::string&name, const std::string& method);
	virtual ~LossLayer();
	virtual void forward();
	virtual void backward();
	virtual void show()const;
	virtual void setup_shape();
	virtual void setup_data();
	std::string get_method(){return method;}
	private:
	void forward_softmax();
	void backward_softmax();
	private:
	std::string method;

	
	Blob softmaxblob, maxs, sums;
	Blob predict, gt;
};

#endif


