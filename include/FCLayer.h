#ifndef FCLAYER_H
#define FCLAYER_H
#include "BaseLayer.h"
class FCLayer: public Layer{
	public:
	FCLayer(const std::string&name, int n, bool is_bias,const std::string& activity);
	virtual ~FCLayer();
	virtual void forward();
	virtual void backward();
	virtual void show()const;
	virtual void setup_shape();
	virtual void setup_data();
	virtual int parameter_number();
	int get_n(){return n;}
	std::string& get_activity(){return activity;}
	Blob weight, weight_dif, bias, bias_dif;
	private:
	int n = 0;
	bool is_bias = 0;
	std::string activity;
};

#endif

