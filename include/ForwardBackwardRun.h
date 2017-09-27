#ifndef FORWARDBACKWARDRUN_H 
#define FORWARDBACKWARDRUN_H 
#include "BaseRun.h"
class InputData{
	public:
	std::string name, path, type, method;
	std::vector<double> mean, std;
	virtual void init() = 0;
	virtual void fill_data(float*& layer_data, int n, int c, int h ,int w, int index) = 0;
	virtual void fill_labels(float*& label_data, int n, int index) = 0;
	virtual std::string& label_name(int label) = 0;
};
class Cifar10Data:public InputData{
	public:
	Cifar10Data();
	~Cifar10Data();
	virtual void init();
	virtual void fill_data(float*& layer_data, int n, int c, int h ,int w, int index);
	virtual void fill_labels(float*& label_data, int n, int index);
	virtual std::string& label_name(int label);
	void load(const std::vector<std::string>& list);
	void load_train();
	void load_test();

	int w = 32, h = 32, c = 3, count=0;
	float *labels = 0, *data = 0;
	std::map<int, std::string> label_map;
	typedef void (Cifar10Data::*FUNC)();
	FUNC f = 0;
};
class ForwardBackwardRun:public Run{
	public:
	ForwardBackwardRun();
	ForwardBackwardRun(const JsonValue& j);
	virtual void show()const;
	virtual void check(const Net004& net)const;
	virtual void operator()(Net004& net, int cur);
	virtual void init(const Net004& net);
	
	InputData* input_data;
	std::map<std::string, std::string> layer_map;

	protected:
	int cur_index = 0;
};
#endif
