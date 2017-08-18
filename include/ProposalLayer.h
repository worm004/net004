#ifndef PROPOSALLAYER_H
#define PROPOSALLAYER_H
#include "BaseLayer.h"
class ProposalLayer: public Layer{
	public:
	ProposalLayer(
		const std::string&name, 
		int feat_stride, 
		const std::vector<std::string>& names,
		const std::string& method);
	virtual ~ProposalLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void setup_dif_shape();
	virtual void setup_dif_data();
	virtual void show()const;

	private:
	int feat_stride;
	std::string method;
};
#endif

