#ifndef FORWARDTESTRUN_H 
#define FORWARDTESTRUN_H 
#include "ForwardBackwardRun.h"
class ForwardTestRun:public ForwardBackwardRun{
	public:
	ForwardTestRun();
	ForwardTestRun(const JsonValue& j);
	virtual void operator()(Net004& net, int cur);
	virtual void init(const Net004& net);

	private:
};
#endif
