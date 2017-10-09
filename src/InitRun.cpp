#include "stdlib.h"
#include "InitRun.h"
#include "Parser.h"
using namespace std;
InitRun::InitRun(){}
InitRun::InitRun(const JsonValue& j):Run(j),generator(rd()){
	if(omit) return;
	const auto& o = j.jobj.at("attrs").jobj;
	method = o.at("method").jv.s;
	if(method == "random") load_random_param();
	else if(method == "model") path = o.at("path").jv.s;
	else{
		printf("unknown method: %s\n",method.c_str());
		exit(0);
	}
}
void InitRun::load_random_param(){
	const auto& o = j_.jobj.at("attrs").jobj;
	if(o.find("layers") == o.end()) return;
	const auto& ls = o.at("layers").jarray;
	for(const auto& l : ls){
		const string& lname = l.jobj.at("name").jv.s.c_str();
		layers[lname];
		for(const auto& b : l.jobj){
			if(b.first == "name") continue;
			const string& bname = b.first;
			layers[lname][bname].init_type = b.second.jobj.at("init_type").jv.s;
			if(b.second.jobj.find("std") != b.second.jobj.end())
				layers[lname][bname].std = b.second.jobj.at("std").jv.d;
			if(b.second.jobj.find("val") != b.second.jobj.end())
				layers[lname][bname].val = b.second.jobj.at("val").jv.d;
		}
	}
}
void InitRun::show_random()const{
	for(const auto& l: layers){
		const string& lname = l.first;
		for(const auto& b: l.second){
			const string& bname = b.first;
			const string& init_type = b.second.init_type;
			if(init_type == "constant"){
				double val = b.second.val;
				printf("    %s %s %s %lf\n",lname.c_str(),bname.c_str(),init_type.c_str(),val);
			}
			else if (init_type == "guassian"){
				double std = b.second.std;
				printf("    %s %s %s %lf\n",lname.c_str(),bname.c_str(),init_type.c_str(),std);
			}
		}
	}
}
void InitRun::show()const{
	if(omit) return;
	Run::show();
	printf("  (layer init)\n");
	if(method == "random") show_random();
	else if(method == "model") printf("  (model) %s\n",path.c_str());
	else{
		printf("unknown method: %s\n",method.c_str());
		exit(0);
	}
}
void InitRun::check(const Net004& net)const{
	if(omit) return;
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
void InitRun::init_constant(Blob& blob, double val){
	//printf("blob is inited by constant val %lf\n",val);
	int n=blob.nchw();
	for(int i=0;i<n;++i) blob.data[i] = val;
}
void InitRun::init_guassian(Blob& blob, double mean, double std){
	//printf("blob is inited by guassian mean %lf, std %lf\n",mean, std);
	normal_distribution<double> distribution(mean,std);
	int n = blob.nchw();
	for(int i=0;i<n;++i)
		blob.data[i] = distribution(generator);
}
void InitRun::init_by_random(Net004& net){
	for(const auto& l:layers){
		const string& lname = l.first;
		Layer* layer = net.ls[lname];
		//printf("%s\n",layer->name.c_str());
		for(const auto& b:l.second){
			const string& bname = b.first;
			Blob& blob = layer->params[bname];
			const string& init_type = b.second.init_type;
			if(init_type == "constant") init_constant(blob,b.second.val);
			else if(init_type == "guassian") init_guassian(blob,b.second.mean,b.second.std);
			//blob.show();
		}
	}
}
void InitRun::init_by_model(Net004& net){
	ModelParser mparser;
	mparser.read_model(path, &net);
}
void InitRun::operator()(Net004& net, int cur){
	if(omit) return;
	printf("[iter %07d] [init] [method %s%s]\n",cur,method.c_str(),(method == "model")?(" "+path).c_str():"");
	if(method == "random") init_by_random(net);
	else if(method == "model") init_by_model(net);
	else{
		printf("unknown method: %s\n",method.c_str());
		exit(0);
	}
}
