#ifndef NETINITRUN_H
#define NETINITRUN_H
#include "BaseRun.h"
struct InitRunUnit{
	public:
	std::string init_type;
	double std;
};
class InitRun:public Run{
	public:
	InitRun();
	InitRun(const JsonValue& j);
	virtual void operator()(Net004& net, int cur);
	virtual void show()const;
	virtual void check(const Net004& net)const;

	std::map<std::string, std::map<std::string, InitRunUnit> > layers;
};

#endif
