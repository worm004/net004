#ifndef NETINITRUN_H
#define NETINITRUN_H
#include <random>
#include "BaseRun.h"
struct InitRunUnit{
	public:
	std::string init_type;
	double std = 0.0, val = 0.0, mean = 0.0;

};
class InitRun:public Run{
	public:
	InitRun();
	InitRun(const JsonValue& j);
	virtual void operator()(Net004& net, int cur);
	virtual void show()const;
	virtual void check(const Net004& net)const;

	void load_random_param();
	void show_random()const;
	void init_constant(Blob& blob, double val);
	void init_guassian(Blob& blob, double mean, double std);
	void init_by_random(Net004& net);
	void init_by_model(Net004& net);

	std::map<std::string, std::map<std::string, InitRunUnit> > layers;
	std::string method, path;

	private:
	std::random_device rd;
	std::default_random_engine generator;
};

#endif
