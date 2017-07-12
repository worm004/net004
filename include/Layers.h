#ifndef LAYERS_H
#define LAYERS_H
#include <string>
#include <vector>
#include <map>

class Layer{
	public:
	Layer(const std::string& name, const std::string& type):name(name),type(type){}
	virtual ~Layer() {}
	virtual void forward() = 0;
	virtual void backward() = 0;
	virtual void show() const = 0;

	protected:
	std::string type;
	std::string name;
};
class ConvLayer: public Layer{
	public:
	ConvLayer(const std::string&name, int filters, int kernel, int stride, int padding, const std::string& activity):
		kernel(kernel), filters(filters), padding(padding), stride(stride), activity(activity), 
		Layer(name,"conv"){}
	virtual ~ConvLayer(){}
	virtual void forward(){}
	virtual void backward(){}
	virtual void show()const {
		printf("[%s%s] name: %s, filters: %d, kernel: %d, stride: %d, padding: %d\n",
			type.c_str(),activity.empty()?"":("+"+activity).c_str(), 
			name.c_str(),
			filters,kernel,stride,padding);
	}

	private:
	int kernel = 0, filters = 0, padding = 0, stride = 0;
	std::string activity;
};
class PoolLayer: public Layer{
	public:
	PoolLayer(const std::string&name, int kernel, int stride, int padding, const std::string& method):
		kernel(kernel),stride(stride),padding(padding),method(method),
		Layer(name,"pool"){}
	virtual ~PoolLayer(){}
	virtual void forward(){}
	virtual void backward(){}
	virtual void show()const {
		printf("[%s%s] name: %s, kernel: %d, stride: %d, padding: %d\n",
			type.c_str(),("+"+method).c_str(), 
			name.c_str(),
			kernel,stride,padding);
	}

	private:
	int stride = 0, padding = 0, kernel = 0;
	std::string method;
};
class LRNLayer: public Layer{
	public:
	LRNLayer(const std::string&name, int n, float beta, float alpha):
		n(n),beta(beta),alpha(alpha),
		Layer(name,"lrn"){}
	virtual ~LRNLayer(){}
	virtual void forward(){}
	virtual void backward(){}
	virtual void show()const {
		printf("[%s] name: %s, n: %d, alpha: %.4f, beta: %.4f\n",
			type.c_str(), 
			name.c_str(),
			n,alpha,beta);
	}
	private:
	int n = 0;
	float beta = 0.0f, alpha = 0.0f;
};
class FCLayer: public Layer{
	public:
	FCLayer(const std::string&name, int n, const std::string& activity):
		n(n), activity(activity),
		Layer(name,"fc"){}
	virtual ~FCLayer(){}
	virtual void forward(){}
	virtual void backward(){}
	virtual void show()const {
		printf("[%s%s] name: %s, n: %d\n",
			type.c_str(),activity.empty()?"":("+"+activity).c_str(), 
			name.c_str(),n);
	}
	private:
	int n = 0;
	std::string activity;
};
class LossLayer: public Layer{
	public:
	LossLayer(const std::string&name, const std::string& method):method(method),Layer(name,"loss"){}
	virtual ~LossLayer(){}
	virtual void forward(){}
	virtual void backward(){}
	virtual void show()const {
		printf("[%s%s] name: %s\n",
			type.c_str(),("+"+method).c_str(), 
			name.c_str());
	}
	private:
	std::string method;
};
class ConcatLayer: public Layer{
	public:
	ConcatLayer(const std::string&name):Layer(name,"concat"){}
	virtual ~ConcatLayer(){}
	virtual void forward(){}
	virtual void backward(){}
	virtual void show()const {
		printf("[%s] name: %s\n", type.c_str(), name.c_str());
	}
};
class Layers{
	public:
	void add_conv(const std::string&name, const std::vector<int>& p4, const std::string& activity);
	void add_pool(const std::string&name, const std::vector<int>& p3, const std::string& method);
	void add_lrn(const std::string&name, int n, float alpha, float beta);
	void add_fc(const std::string&name, int n, const std::string& activity);
	void add_loss(const std::string&name, const std::string& method);
	void add_concat(const std::string&name);
	void add(const std::string& name, Layer** p);
	void show();
	void show(const std::string& name);
	void clear();
	Layer* operator [] (const std::string& name);
	bool exist(const std::string& name);
	int count();

	private:
	std::map<std::string, Layer*> layers;
};
#endif
