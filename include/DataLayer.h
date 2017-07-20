#ifndef DATALAYER_H
#define DATALAYER_H
#include "BaseLayer.h"
class DataLayer: public Layer{
	public:
	DataLayer(
		const std::string&name,
		int n, 
		int c, 
		int h, 
		int w,
		const std::string& method);
	virtual ~DataLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void show() const;
	void add_image(unsigned char* data, int index);

	std::string method;
	int n = 0, c = 0, h = 0, w = 0;
};
#endif
