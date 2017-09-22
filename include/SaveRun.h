#ifndef SAVERUN_H
#define SAVERUN_H
#include "BaseRun.h"
class SaveRun:public Run{
	public:
	SaveRun();
	SaveRun(const RunUnit& u);
	std::string dir;
	virtual void operator()(Net004& net, int cur);
};
#endif
