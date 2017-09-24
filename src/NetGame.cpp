#include "stdlib.h"
#include "NetGame.h"
#include "BaseRun.h"
#include "Parser.h"
#include "SaveRun.h"
#include "DisplayRun.h"
#include "UpdateRun.h"
#include "InitRun.h"
#include "ForwardBackwardRun.h"
#include "ForwardTestRun.h"

using namespace std;
template<typename T> 
Run* create_run(const JsonValue& u){
	return new T(u);
}
NetGame::NetGame(){
	run_type_map = {
		{"save",&create_run<SaveRun>},
		{"display",&create_run<DisplayRun>},
		{"update",&create_run<UpdateRun>},
		{"init",&create_run<InitRun>},
		{"forward_backward",&create_run<ForwardBackwardRun>},
		{"forward_test",&create_run<ForwardTestRun>}
	};
}
void NetGame::load(const std::string& path){
	JsonParser gparser;	
	gparser.read(path);
	const auto& o = gparser.j.jobj;
	name = o.at("name").jv.s;
	type = o.at("type").jv.s;
	net_path =  o.at("net_path").jv.s;
	batch_size = o.at("batch_size").jv.d;
	max_iter =  o.at("max_iter").jv.d;
	for(const auto& run:o.at("runs").jarray){
		const auto& attrs = run.jobj.at("attrs").jobj;
		const auto& name = attrs.at("name").jv.s;
		runs[name] = run_type_map[attrs.at("type").jv.s](run);
		//runs[name]->show();
	}

	JsonParser nparser;	
	nparser.read(net_path);
	net.name = nparser.j.jobj["net_name"].jv.s;
	if(nparser.j.jobj.find("layers") == nparser.j.jobj.end()){
		printf("no layers found\n");
		exit(0);
	}
	for(int i=0;i<nparser.j.jobj["layers"].jarray.size();++i){
		net.ls.add(nparser.j.jobj["layers"].jarray[i]);
	}
	net.ls.init();
	for(const auto&run:runs) run.second->check(net);
}
void NetGame::run(){
	for(int iter = 0;iter<max_iter;++iter)
		for(const auto& i:runlist){
			if(i.)
			runs[i]->operator()(net,iter);
		}
}
void NetTrain::init(){
	runlist = {"train_step","update","display","save","test_step"};
	runs["init"]->operator()(net,0);
}
