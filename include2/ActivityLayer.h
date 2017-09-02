#ifndef ACTIVITYLAYER_H
#define ACTIVITYLAYER_H
#include "BaseLayer.h"
class ActivityLayer: public Layer{
	public:
	ActivityLayer();
	ActivityLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();

	float neg_slope;
	std::string method;
};
#endif
