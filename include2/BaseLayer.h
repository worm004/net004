#ifndef BASELAYER_H
#define BASELAYER_H
#include <map>
#include <string>
#include <vector>
#include "Blob.h"
struct ParamUnit{
	ParamUnit();
	ParamUnit(float v);
	ParamUnit(const std::string& v);
	std::string type;
	float fval;
	std::string sval;
};
struct LayerUnit{
	void geta(const std::string& key, float& val) const;
	void geta(const std::string& key, std::string& val) const;
	void clear();
	bool exista(const std::string&key) const;
	bool existp(const std::string&key) const;
	bool existi(const std::string&key) const;
	void checka(const std::string& key, const std::string& type) const;
	std::map<std::string, ParamUnit> attrs;
	std::map<std::string, std::vector<int>> params;
	std::map<std::string, int> inputs;
};

int i2o_floor(int w, int kernel, int stride, int padding);
int i2o_ceil(int w, int kernel, int stride, int padding);
class Layer{
	public:
	Layer();
	Layer(const LayerUnit& u);
	virtual ~Layer();
	void set_inplace(bool inplace);
	virtual void show();
	virtual void setup_outputs() = 0;

	public:
	std::string name, type;
	std::vector<Blob> inputs, outputs;
	std::map<std::string, Blob> params;

	LayerUnit u;
	bool inplace = false;
};
#endif
