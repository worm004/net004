#ifndef FCLAYER_H
#define FCLAYER_H
#include "BaseLayer.h"
class FCLayer: public Layer{
	public:
	FCLayer();
	FCLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();

	bool bias;
	int num;
};
#endif
