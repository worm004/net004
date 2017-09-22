#ifndef UPDATERUN_H
#define UPDATERUN_H
#include "BaseRun.h"
class UpdateRun:public Run{
	public:
	UpdateRun();
	UpdateRun(const JsonValue& j);
	std::string solver;
	double momentum, weight_decay, base_lr;
	virtual void operator()(Net004& net, int cur);
};
#endif
