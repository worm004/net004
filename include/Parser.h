#ifndef PARSER_H
#define PARSER_H
#include <vector>
#include <string>
#include <map>
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
class NetParser{
	public:
	void read_net(const std::string& path);
	void write_net(const std::string& path);

	void add_layer(const LayerUnit& u);
	void set_net_name(const std::string& name);
	std::string get_net_name();
	const std::vector<LayerUnit>& get_layers();
	
	private:
	std::vector<LayerUnit> layers;
	std::string net_name;
};
class Net004;
class ModelParser{
	public:
	ModelParser();
	void read_model(const std::string& path, Net004* net);
};
#endif
