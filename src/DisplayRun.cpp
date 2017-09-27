#include "stdlib.h"
#include "DisplayRun.h"
DisplayRun::DisplayRun(){}
DisplayRun::DisplayRun(const JsonValue& j):Run(j){
	if(omit) return;
	const auto& a = j.jobj.at("attrs").jobj.at("layers").jarray;
	for(const auto& i: a) layers.push_back(i.jv.s);
}
void DisplayRun::show()const{
	if(omit) return;
	Run::show();
	printf("  (layers) ");
	for(int i=0;i<layers.size();++i){
		if(i != 0) printf(", ");
		printf("%s",layers[i].c_str());
	}
	printf("\n");
}
void DisplayRun::check(const Net004& net)const{
	if(omit) return;
	Run::check(net);
	for(const auto& i: layers)
		if (net.ls.n2i.find(i) == net.ls.n2i.end()){
			printf("cannot find layer: %s\n",i.c_str());
			exit(0);
		}
}
void DisplayRun::operator()(Net004& net, int cur){
	if(omit) return;
	if(cur%iter_interval != 0) return;
	for(auto ln:layers){
		Layer* layer = net[ln];
		for(int i=0;i<layer->outputs.size();++i){
			int chw = layer->outputs[i].chw();
			float *data = layer->outputs[i].data;
			int batch_size = layer->outputs[i].n;
			for(int b=0;b<batch_size;++b){
				printf("[iter %d] [display] [layer %s] [output %d] [batch %d]:",cur,ln.c_str(),i,b);
				for(int j=0;j<chw;++j)
					printf(" %.2f",data[j+b*chw]);
				printf("\n");
			}
		}
	}
}
