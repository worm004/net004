#ifndef FILLANDTESTRUN_H
#define FILLANDTESTRUN_H
#include "FillAndForwardRun.h"
class FillAndTestRun:public FillAndForwardRun{
	public:
	FillAndTestRun();
	FillAndTestRun(const RunUnit& u);
	virtual void operator()(Net004& net, int cur);
};
#endif
