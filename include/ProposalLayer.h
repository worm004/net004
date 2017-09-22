#ifndef PROPOSALLAYER_H
#define PROPOSALLAYER_H
#include "BaseLayer.h"
class ProposalLayer: public Layer{
	public:
	ProposalLayer();
	ProposalLayer(const JsonValue& j);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	void generate_anchors();
	int feat_stride;

	int RPN_PRE_NMS_TOP_N = 6000/*test phase*/, 
		RPN_POST_NMS_TOP_N = 300/*test phase*/, 
		RPN_MIN_SIZE = 16/*test phase*/;
	float RPN_NMS_THRESH = 0.7/*test phase*/;
	std::vector<float> anchor_scales = {8,16,32}, 
		ratios = {0.5f,1.0f,2.0f};
	int base_size = 16;
	std::vector<std::vector<float> > anchors;
};
#endif
