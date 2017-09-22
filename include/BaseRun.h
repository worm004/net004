#ifndef BASERUN_H
#define BASERUN_H
#include "Net004.h"
#include "JsonParser.h"
class Run{
	public:
	Run();
	Run(const JsonValue& j);

	std::string type;
	int iter;
	virtual void operator()(Net004& net, int cur) = 0;
	JsonValue j_;
};
#endif
