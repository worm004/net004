#ifndef CROPLAYER_H
#define CROPLAYER_H
#include "BaseLayer.h"
class CropLayer: public Layer{
	public:
	CropLayer();
	CropLayer(const JsonValue& j);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	int axis, offset;
	std::vector<int> offsets;
};
#endif
