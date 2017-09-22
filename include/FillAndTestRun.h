#ifndef FILLANDTESTRUN_H
#define FILLANDTESTRUN_H
#include "FillAndForwardRun.h"
class FillAndTestRun:public FillAndForwardRun{
	public:
	FillAndTestRun();
	FillAndTestRun(const JsonValue& j);
	virtual void operator()(Net004& net, int cur);
};
#endif
