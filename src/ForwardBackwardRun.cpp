#include "stdlib.h"
#include "ForwardBackwardRun.h"
using namespace std;
ForwardBackwardRun::ForwardBackwardRun(){ }
ForwardBackwardRun::ForwardBackwardRun(const JsonValue& j):Run(j){
	const auto& odata = j.jobj.at("attrs").jobj.at("data").jobj;
	data.name = odata.at("name").jv.s;
	data.type = odata.at("type").jv.s;
	data.path = odata.at("path").jv.s;
	const auto& omean = odata.at("mean").jarray;
	for(const auto& i:omean)
		data.mean.push_back(i.jv.d);
	const auto& ostd = odata.at("std").jarray;
	for(const auto& i:ostd)
		data.std.push_back(i.jv.d);
	const auto& omap = j.jobj.at("attrs").jobj.at("layer_map").jobj;
	for(const auto&i: omap)
		layer_map[i.first] = i.second.jv.s;

}
void ForwardBackwardRun::show()const{
	Run::show();
	printf("  (data)\n");
	printf("    (name) %s\n",data.name.c_str());
	printf("    (type) %s\n",data.type.c_str());
	printf("    (path) %s\n",data.path.c_str());
	printf("    (mean) ");
	for(int i=0;i<data.mean.size();++i){
		if(i!=0)printf(", ");
		printf("%lf",data.mean[i]);
	}
	printf("\n");
	printf("    (std) ");
	for(int i=0;i<data.std.size();++i){
		if(i!=0)printf(", ");
		printf("%lf",data.std[i]);
	}
	printf("\n");
}
void ForwardBackwardRun::check(const Net004& net)const{
	Run::check(net);
	for(const auto& l: layer_map){
		const string& lname = l.second;
		if(net.ls.n2i.find(lname)==net.ls.n2i.end()){
			printf("cannot find layer: %s\n",lname.c_str());
			exit(0);
		}
	}
}
void ForwardBackwardRun::operator()(Net004& net, int cur) {
	if(cur%iter_interval != 0) return;
	printf("[%d]run: train step\n",cur);
}
