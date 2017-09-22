#include "stdlib.h"
#include "NetGame.h"
#include "BaseRun.h"
#include "Parser.h"
#include "FillAndForwardRun.h"
#include "NetInitRun.h"
#include "UpdateRun.h"
#include "DisplayRun.h"
#include "FillAndTestRun.h"
#include "SaveRun.h"

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
		{"fill_and_test",&create_run<FillAndTestRun>},
		{"net_init",&create_run<NetInitRun>},
		{"fill_and_forward",&create_run<FillAndForwardRun>}
	};
}
void NetGame::load(const std::string& path){
	//GameParser gparser;
	//gparser.read(path);
	//name = gparser.name;
	//type = gparser.type;
	//net_path = gparser.net_path;
	//batch_size = gparser.batch_size;
	//max_iter = gparser.max_iter;
	//for(const auto& run:gparser.runs)
	//	runs[run.type] = run_type_map[run.type](run);
}
void NetGame::run(){
	for(const auto& i : runlist)
		if(runs.find(i) == runs.end()){
			printf("ERROR: not find run %s\n",i.c_str());
			exit(0);
		}
	for(int iter = 0;iter<max_iter;++iter)
		for(const auto& i:runlist)
			runs[i]->operator()(net,iter);
}
void NetTrain::init(){
	runlist = {"fill_and_forward","update","display","save","fill_and_test"};
	if(runs.find("init_layer_params") == runs.end()) return;
	runs["init_layer_params"]->operator()(net,0);
}
