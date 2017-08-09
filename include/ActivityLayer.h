#ifndef ACTIVITYLAYER_H
#define ACTIVITYLAYER_H
#include "BaseLayer.h"

class ActivityLayer: public Layer{
	public:
	ActivityLayer(const std::string& name, const std::string& method);
	virtual ~ActivityLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void setup_dif_shape();
	virtual void setup_dif_data();
	virtual void show() const;

	void forward_relu();
	std::string get_method(){return method;}

	private:
	std::string method;
	bool *mask = 0;
};

#endif
