#ifndef FCLAYER_H
#define FCLAYER_H
#include "BaseLayer.h"
class FCLayer: public Layer{
	public:
	FCLayer();
	FCLayer(const JsonValue& j);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	virtual void backward();

	bool bias;
	int num;
};
#endif
