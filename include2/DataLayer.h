#ifndef DATALAYER_H
#define DATALAYER_H
#include "BaseLayer.h"
class DataLayer: public Layer{
	public:
	DataLayer();
	DataLayer(const LayerUnit& u);

	int n,c,h,w;
};
#endif
