#ifndef NETINITRUN_H
#define NETINITRUN_H
#include "BaseRun.h"
class NetInitRun:public Run{
	public:
	NetInitRun();
	NetInitRun(const RunUnit& u);
	virtual void operator()(Net004& net, int cur);
};

#endif
