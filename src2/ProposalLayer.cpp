#include <cmath>
#include "stdlib.h"
#include "ProposalLayer.h"
using namespace std;
ProposalLayer::ProposalLayer(){}
ProposalLayer::ProposalLayer(const LayerUnit& u):Layer(u){
	float v;
	u.geta("feat_stride",v); feat_stride = v;
	generate_anchors();
}
void ProposalLayer::show(){
	Layer::show();
	printf("  (feat_stride) %d\n", feat_stride);
}
void ProposalLayer::setup_outputs(){
	outputs[0].set_shape(inputs[0].n,RPN_POST_NMS_TOP_N*5,1,1);
	setup_outputs_data();
}
void ProposalLayer::generate_anchors(){
	anchors.clear();
	float w0 = base_size, 
		h0 = base_size,
		x_ctr = 0.5f * (w0 - 1),
		y_ctr = 0.5f * (h0 - 1);

	for(int i=0;i<ratios.size();++i){
		float ss = w0 * h0 / ratios[i];
		int w1 = int(sqrt(ss)+0.5f),
		    h1 = int(w1*ratios[i]+0.5f);
		for(int j=0;j<anchor_scales.size();++j){
			float w2 = w1 * anchor_scales[j],
				h2 = h1 * anchor_scales[j];
			float l2 = x_ctr - 0.5f*(w2-1),
			      t2 = y_ctr - 0.5f*(h2-1),
			      r2 = x_ctr + 0.5f*(w2-1),
			      b2 = y_ctr + 0.5f*(h2-1);
			//printf("%g %g %g %g\n",l2,t2,r2,b2);
			anchors.push_back({l2,t2,r2,b2});
		}
	}
}
void nms2(vector<vector<float> >& rs, float T,int top_n){
	int count = rs.size();
	if(count == 0) return;
	vector<int> indics(count);
	for(int i=0;i<count;++i) indics[i] = i;
	sort(indics.begin(), indics.end(),
		[&](int i, int j) {return rs[i][4] > rs[j][4];});

	vector<vector<float> > temp;
	count = min(top_n,count);
	vector<bool> deletes(count,false);
	for(int i=0;i<count;++i){
		int index = indics[i];
		temp.push_back({rs[index][0], rs[index][1], rs[index][2], rs[index][3], 
				(rs[index][2]-rs[index][0]+1) * (rs[index][3] - rs[index][1] + 1), 
				rs[index][4]});
		//printf("+ %d %f %f %f %f %f\n",i,rs[index][0], rs[index][1], rs[index][2], rs[index][3],rs[index][4]);
	}

	rs.clear();
	for(int i=0;i<count;++i){
		if(deletes[i]) continue;
		rs.push_back({temp[i][0], temp[i][1], temp[i][2], temp[i][3],temp[i][5]});
		for(int j=i+1;j<count;++j){
			if(deletes[j]) continue;
			float l = std::max(temp[i][0], temp[j][0]),
			      t = std::max(temp[i][1], temp[j][1]),
			      r = std::min(temp[i][2], temp[j][2]),
			      b = std::min(temp[i][3], temp[j][3]),
			      area = max(0.0f,(r-l+1)) * max(0.0f,(b-t+1));
			if (area / (temp[i][4] + temp[j][4] - area) < T)
				continue;
			deletes[j] = true;
		}
	}
	//for(int i=0;i<rs.size();++i)
	//	printf("+ %d %f %f %f %f %f\n",i,rs[i][0], rs[i][1], rs[i][2], rs[i][3],rs[i][4]);
}
void ProposalLayer::forward(){
	//show_inputs();
	int n = inputs[0].n, 
		h = inputs[0].h, 
		w = inputs[0].w, 
		a = anchors.size(), 
		hw = h*w;
	float* sdata = inputs[0].data + hw*a,
		*ddata = inputs[1].data,
		*odata = outputs[0].data,
		*idata = inputs[2].data;
	float min_size = RPN_MIN_SIZE * idata[2];
	memset(odata,0,sizeof(float)*outputs[0].nchw());
	for(int b = 0;b<n;++b){
		vector<vector<float>> temp;
		for(int i=0, shift_y=0,loc=0;i<h;++i,shift_y+=feat_stride)
		for(int j=0, shift_x = 0;j<w;++j,++loc,shift_x+=feat_stride){
			for(int k = 0;k<a;++k){
				int dindex = hw*4*k+loc;
				float aw = anchors[k][2] - anchors[k][0] + 1,
					ah = anchors[k][3] - anchors[k][1] + 1,
					pcx = ddata[dindex] * aw + anchors[k][0] + shift_x + 0.5f*aw,
					pcy = ddata[dindex+hw] * ah + anchors[k][1] + shift_y + 0.5f*ah,
					pw = exp(ddata[dindex + hw * 2]) * aw,
					ph = exp(ddata[dindex + hw * 3]) * ah,
					score = sdata[hw*k+loc];
				
				float l = std::max(pcx - 0.5f * pw, 0.0f), 
					t = std::max(pcy - 0.5f * ph, 0.f), 
					r = std::min(pcx + 0.5f * pw, idata[1]-1),
					b = std::min(pcy + 0.5f * ph, idata[0]-1);
				
				if((r-l+1 >= min_size) && (b-t+1 >= min_size))
					temp.push_back({l,t,r,b,score});
			}
		}
		nms2(temp,RPN_NMS_THRESH,RPN_PRE_NMS_TOP_N);
		int selectn = std::min(int(temp.size()),RPN_POST_NMS_TOP_N);
		for(int i=0;i<selectn;++i){
			odata[i*5] = temp[i][4];
			odata[i*5+1] = temp[i][0];
			odata[i*5+2] = temp[i][1];
			odata[i*5+3] = temp[i][2];
			odata[i*5+4] = temp[i][3];
		}
		sdata += inputs[0].chw();
		ddata += inputs[1].chw();
		odata += 5*RPN_POST_NMS_TOP_N;
	}
	//show_outputs();
}
