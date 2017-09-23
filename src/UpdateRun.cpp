#include "stdlib.h"
#include "UpdateRun.h"
using namespace std;
UpdateRun::UpdateRun(){}
UpdateRun::UpdateRun(const JsonValue& j):Run(j){
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
}
void UpdateRun::show()const{
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
}
