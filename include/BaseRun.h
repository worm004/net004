#ifndef BASERUN_H
#define BASERUN_H
#include "Net004.h"
struct RunLayerParam{
	std::vector<std::string> layers;
	std::map<std::string, std::map<std::string, ParamUnit> > params;
};
struct RunUnit{
	std::string type;
	int iter = 1;
	std::map<std::string, ParamUnit> attrs;
	std::vector<std::string> add_layers;
};
class Run{
	public:
	Run();
	Run(const RunUnit& u);

	std::string type;
	int iter;
	virtual void operator()(Net004& net, int cur) = 0;
};
#endif
