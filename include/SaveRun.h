#ifndef SAVERUN_H
#define SAVERUN_H
#include "BaseRun.h"
class SaveRun:public Run{
	public:
	SaveRun();
	SaveRun(const JsonValue& j);
	virtual void operator()(Net004& net, int cur);
	virtual void show()const;
	virtual void check(const Net004& net)const;

	std::string dir;
};
#endif
