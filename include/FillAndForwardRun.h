#ifndef FILLANDFORWARD_H
#define FILLANDFORWARD_H
#include "BaseRun.h"
struct InputData{
	std::string path, type;//image/label list
	std::vector<std::string> imgs;
	std::vector<int> labels;
	std::string img_layer, label_layer;
};
class FillAndForwardRun:public Run{
	public:
	FillAndForwardRun();
	FillAndForwardRun(const RunUnit& u);
	
	InputData data;
	virtual void operator()(Net004& net, int cur);
};
#endif
