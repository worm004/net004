#include "stdlib.h"
#include "UpdateRun.h"
using namespace std;
UpdateRun::UpdateRun(){}
UpdateRun::UpdateRun(const JsonValue& j):Run(j){
	if(omit) return;
	const auto& o = j.jobj.at("attrs").jobj;
	solver = o.at("solver").jv.s;
	momentum = o.at("momentum").jv.d;
	weight_decay = o.at("weight_decay").jv.d;

	const auto& o_lr = o.at("lr").jobj;
	lr.type = o_lr.at("type").jv.s;
	lr.base = o_lr.at("base").jv.d;

	if(o_lr.find("mults")!=o_lr.end()){
		const auto& a_lr_mult = o_lr.at("mults").jarray;
		for(const auto& i:a_lr_mult){
			const string& layer_name = i.jobj.at("name").jv.s;
			lr.mults[layer_name];
			for(const auto& jj:i.jobj){
				if(jj.first == "name") continue;
				const string& blob_name = jj.first;
				double mult = jj.second.jv.d;
				lr.mults[layer_name][blob_name] = mult;
			}
		}
	}
	f = &UpdateRun::init;
	if(solver == "sgd") update_f = &UpdateRun::sgd;
	else{
		printf("unknown solver type: %s\n",solver.c_str());
		exit(0);
	}
}
void UpdateRun::show()const{
	if(omit) return;
	Run::show();
	printf("  (solver) %s\n", solver.c_str());
	printf("  (momentum) %lf\n", momentum);
	printf("  (weight_decay) %lf\n", weight_decay);
	printf("  (learning_rate) %s %lf\n", lr.type.c_str(), lr.base);
	printf("  (mults)\n");
	for(const auto& ll: lr.mults){
		const string& l = ll.first;
		for(const auto& bb: ll.second){
			const string& b = bb.first;
			double mult = bb.second;
			printf("    %s %s %lf\n",l.c_str(),b.c_str(),mult);
		}
	}
}
void UpdateRun::check(const Net004& net)const{
	if(omit) return;
	Run::check(net);
	for(const auto& l:lr.mults){
		const string& lname = l.first;
		if(net.ls.n2i.find(lname)==net.ls.n2i.end()){
			printf("cannot find layer: %s\n",lname.c_str());
			exit(0);
		}
		Layer* layer = net.ls.layers[net.ls.n2i.at(lname)];
		for(const auto& b:l.second){
			if(layer->params.find(b.first)==layer->params.end()){
				printf("cannot find param %s in layer %s\n",b.first.c_str(),lname.c_str());
				exit(0);
			}
		}
	}

}
void UpdateRun::operator()(Net004& net, int cur){
	if(omit) return;
	(this->*f)(net,cur);
}
void UpdateRun::init(Net004& net, int cur){
	for(int i=0;i<net.ls.size();++i){
		Layer* l = net.ls[i];
		if(l->params.size()){
			acc_diff[l->name];
			history[l->name];
		}
		for(const auto&p:l->params){
				Blob& b = acc_diff[l->name][p.first];
				b.set_shape(p.second);
				b.alloc();
				memset(b.data,0,sizeof(float)*b.nchw());
		}
		for(const auto&p:l->params){
				Blob& b = history[l->name][p.first];
				b.set_shape(p.second);
				b.alloc();
				memset(b.data,0,sizeof(float)*b.nchw());
		}
	}
	f = &UpdateRun::update;
	(this->*f)(net,cur);
}
void UpdateRun::accumulate(Net004& net, int cur){
	for(int i=0;i<net.ls.size();++i){
		Layer* l = net.ls[i];
		for(const auto&p:l->diff_params){
			Blob& acc = acc_diff[l->name][p.first];
			const Blob& src = p.second;

			float *acc_data = (float*)acc.data,
				*src_data = (float*)src.data;
			int nchw = acc.nchw();
			for(int j=0;j<nchw;++j)
				acc_data[j] += src_data[j];
		}
	}
}
void UpdateRun::sgd(Net004& net, int cur){
	// generate learning rate
	float learning_rate = lr.base;
	if(lr.type == "fixed") learning_rate *= 1.0f;
	printf("[iter %07d] [update] [solver %s] [learning_rate %g]\n",cur,solver.c_str(),learning_rate);

	// normalize
	if(iter_interval > 1){
		float normalize = 1.0f/float(iter_interval);
		for(auto& l:acc_diff)
			for(auto& b:l.second){
				int nchw = b.second.nchw();
				float *data = b.second.data;
				for(int i=0;i<nchw;++i)
					data[i]*=normalize;
			}
	}

	// regularize
	for(auto& l:acc_diff)
		for(auto& b:l.second){
			int nchw = b.second.nchw();
			float *diff_data = b.second.data,
			      *data = net[l.first]->params[b.first].data;
			for(int i=0;i<nchw;++i)
				diff_data[i] += weight_decay*data[i];
		}
	// update
	for(auto& l:acc_diff)
		for(auto& b:l.second){
			float local_lr = learning_rate * lr.mults[l.first][b.first];
			int nchw = b.second.nchw();

			Blob & history_blob = history[l.first][b.first],
			     & param_blob = net[l.first]->params[b.first];

			float *diff_data = b.second.data,
			      *history_data = history_blob.data,
			      *data = param_blob.data;
			for(int i=0;i<nchw;++i){
				history_data[i] = history_data[i] * momentum + diff_data[i]*local_lr;
				// history = history * momentum + lr * (weight_decay * data + diff)
				// data -= d * momentum + lr * (weight_decay * data + diff)
				data[i] -= history_data[i];
			}
		}
	
	for(auto& l:acc_diff)
		for(auto& b:l.second){
			int nchw = b.second.nchw();
			memset(b.second.data,0,nchw*sizeof(float));
		}
}
void UpdateRun::update(Net004& net, int cur){
	accumulate(net, cur);
	if(cur%iter_interval != 0) return;
	(this->*update_f)(net, cur);
}
