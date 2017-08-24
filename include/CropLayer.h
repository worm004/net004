#ifndef CROPLAYER_H
#define CROPLAYER_H
#include "BaseLayer.h"
class CropLayer: public Layer{
	public:
	CropLayer(
		const std::string&name, 
		int axis, 
		const std::vector<int>& offset, 
		const std::vector<std::string>& names);
	virtual ~CropLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void setup_dif_shape();
	virtual void setup_dif_data();
	virtual void show()const;

	private:
	int axis;
	std::vector<int> offset;
};
#endif

