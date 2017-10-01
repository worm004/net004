#ifndef BASELAYER_H
#define BASELAYER_H
#include <map>
#include <string>
#include <vector>
#include "Blob.h"
#include "JsonParser.h"

int i2o_floor(int w, int kernel, int stride, int padding);
int i2o_ceil(int w, int kernel, int stride, int padding);
class Layer{
	public:
	Layer();
	Layer(const JsonValue& j);
	virtual ~Layer();
	void set_inplace(bool inplace);
	void show_inputs();
	void show_outputs();
	virtual void show();
	virtual void setup_outputs() = 0;
	virtual void forward() = 0;
	virtual void backward();
	void setup_outputs_data();
	bool is_inplace();
	virtual void init_train();

	public:
	std::string name, type;
	// test
	std::vector<Blob> inputs, outputs;
	std::map<std::string, Blob> params;

	// train
	std::vector<Blob> diff_inputs, diff_outputs;
	std::map<std::string, Blob> diff_params;

	JsonValue j_;
	bool inplace = false;
	bool train = false;
};
#endif
