#ifndef ACTIVITYLAYER_H
#define ACTIVITYLAYER_H
#include "BaseLayer.h"
class ActivityLayer: public Layer{
	public:
	ActivityLayer();
	ActivityLayer(const JsonValue& j);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	void forward_relu();
	float neg_slope;
	std::string method;
	typedef void (ActivityLayer::*FORWARD_FUNC)();
	FORWARD_FUNC f = 0;
};
#endif
