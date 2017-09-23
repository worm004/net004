#ifndef FORWARDBACKWARDRUN_H 
#define FORWARDBACKWARDRUN_H 
#include "BaseRun.h"
struct InputData{
	std::string name, path, type;//image/label list
	std::vector<double> mean, std;
	std::vector<std::string> imgs;
	std::vector<int> labels;
};
class ForwardBackwardRun:public Run{
	public:
	ForwardBackwardRun();
	ForwardBackwardRun(const JsonValue& j);
	virtual void show()const;
	virtual void check(const Net004& net)const;
	virtual void operator()(Net004& net, int cur);
	
	InputData data;
	std::map<std::string, std::string> layer_map;
};
#endif
