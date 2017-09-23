#include "stdlib.h"
#include "InitRun.h"
using namespace std;
InitRun::InitRun(){}
InitRun::InitRun(const JsonValue& j):Run(j){
	const auto& o = j.jobj.at("attrs").jobj;
	if(o.find("layers") != o.end()){
		const auto& ls = o.at("layers").jarray;
		for(const auto& l : ls){
			const string& lname = l.jobj.at("name").jv.s.c_str();
			layers[lname];
			for(const auto& b : l.jobj){
				if(b.first == "name") continue;
				const string& bname = b.first;
				layers[lname][bname].init_type = b.second.jobj.at("init_type").jv.s;
				if(b.second.jobj.find("std") != b.second.jobj.end()){
					layers[lname][bname].std = b.second.jobj.at("std").jv.d;
				}
			}
		}
	}
}
void InitRun::operator()(Net004& net, int cur){
}
void InitRun::show()const{
	Run::show();
	printf("  (layer init)\n");
	for(const auto& l: layers){
		const string& lname = l.first;
		for(const auto& b: l.second){
			const string& bname = b.first;
			const string& init_type = b.second.init_type;
			if(init_type == "constant")
				printf("    %s %s %s\n",lname.c_str(),bname.c_str(),init_type.c_str());
			else if (init_type == "guassian"){
				double std = b.second.std;
				printf("    %s %s %s %lf\n",lname.c_str(),bname.c_str(),init_type.c_str(),std);
			}
		}
	}
}
void InitRun::check(const Net004& net)const{
	Run::check(net);
	for(const auto& l:layers){
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
