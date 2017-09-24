#include "stdlib.h"
#include "DisplayRun.h"
DisplayRun::DisplayRun(){}
DisplayRun::DisplayRun(const JsonValue& j):Run(j){
	const auto& a = j.jobj.at("attrs").jobj.at("layers").jarray;
	for(const auto& i: a) layers.push_back(i.jv.s);
}
void DisplayRun::show()const{
	Run::show();
	printf("  (layers) ");
	for(int i=0;i<layers.size();++i){
		if(i != 0) printf(", ");
		printf("%s",layers[i].c_str());
	}
	printf("\n");
}
void DisplayRun::check(const Net004& net)const{
	Run::check(net);
	for(const auto& i: layers)
		if (net.ls.n2i.find(i) == net.ls.n2i.end()){
			printf("cannot find layer: %s\n",i.c_str());
			exit(0);
		}
}
void DisplayRun::operator()(Net004& net, int cur){
	printf("[%d]run: display\n",cur);
}
