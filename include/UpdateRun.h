#ifndef UPDATERUN_H
#define UPDATERUN_H
#include "BaseRun.h"
struct LearningRate{
	std::string type;
	double base;
	std::map<std::string, std::map<std::string,double> > mults;
};
class UpdateRun:public Run{
	public:
	UpdateRun();
	UpdateRun(const JsonValue& j);
	virtual void operator()(Net004& net, int cur);
	virtual void show()const;
	virtual void check(const Net004& net)const;

	private:
	std::string solver;
	double momentum, weight_decay;
	LearningRate lr;
};
#endif
