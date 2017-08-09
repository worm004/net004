#ifndef LRNLAYER_H
#define LRNLAYER_H
#include "BaseLayer.h"
class LRNLayer: public Layer{
	public:
	LRNLayer(const std::string&name, int n, float alpha, float beta);
	virtual ~LRNLayer();
	virtual void forward();
	virtual void backward();
	virtual void show()const;
	virtual void setup_shape();
	virtual void setup_data();
	virtual void setup_dif_shape();
	virtual void setup_dif_data();

	private:
	int n = 0;
	float beta = 0.0f, alpha = 0.0f;
	Blob buffer;
};

#endif

