#ifndef ROIPOOLINGLAYER_H
#define ROIPOOLINGLAYER_H
#include "BaseLayer.h"
class ROIPoolingLayer: public Layer{
	public:
	ROIPoolingLayer(
		const std::string&name, 
		int h, int w, float scal, const std::vector<std::string>& names);
	virtual ~ROIPoolingLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void setup_dif_shape();
	virtual void setup_dif_data();
	virtual void show()const;

	private:
	int h,w;
	float scale;
};
#endif

