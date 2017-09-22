#ifndef NETINITRUN_H
#define NETINITRUN_H
#include "BaseRun.h"
class NetInitRun:public Run{
	public:
	NetInitRun();
	NetInitRun(const JsonValue& j);
	virtual void operator()(Net004& net, int cur);
};

#endif
