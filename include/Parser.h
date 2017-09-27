#ifndef PARSER_H
#define PARSER_H
#include <vector>
#include <string>
#include <map>
class Net004;
class ModelParser{
	public:
	ModelParser();
	void read_model(const std::string& path, Net004* net);
	void write_model(const std::string& path, Net004* net);
};
#endif
