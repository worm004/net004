#ifndef DISPLAYRUN_H
#define DISPLAYRUN_H
#include "BaseRun.h"
struct DisplayRun:public Run{
	public:
	DisplayRun();
	DisplayRun(const RunUnit& u);
	virtual void operator()(Net004& net, int cur);
};
#endif