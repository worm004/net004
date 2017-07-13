#ifndef BASELAYER_H
#define BASELAYER_H
#include <cmath>
#include <string>
#include <vector>
#include "Blob.h"

class Layer{
	public:
	Layer(const std::string& name, const std::string& type);
	virtual ~Layer();
	virtual void forward() = 0;
	virtual void backward() = 0;
	virtual void show() const = 0;
	virtual void setup_shape();
	virtual void setup_data();
	virtual void connect2(Layer& l);
	virtual int parameter_number();
	void setup();
	static int i2o_floor(int w, int kernel, int stride, int padding){
		return (w + 2 * padding - kernel) / stride + 1;
	}
	static int i2o_ceil(int w, int kernel, int stride, int padding){
		return static_cast<int>(std::ceil(static_cast<float>( w + 2 * padding - kernel) / stride)) + 1;
	}

	std::vector<Blob> inputs, input_difs, outputs, output_difs;
	std::string type, name;
};
#endif
