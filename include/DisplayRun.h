#ifndef DISPLAYRUN_H
#define DISPLAYRUN_H
#include "BaseRun.h"
struct DisplayRun:public Run{
	public:
	DisplayRun();
	DisplayRun(const JsonValue& j);
	virtual void operator()(Net004& net, int cur);
	virtual void show()const;
	virtual void check(const Net004& net)const;

	private:
	std::vector<std::string> layers;
};
#endif
