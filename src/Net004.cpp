#include "stdlib.h"
#include <string>
#include <map>
#include <iostream>
#include "Net004.h"
using namespace std;

void Net004::check(){
	if(cs.sorted_cs.size() != ls.count()){
		printf("error: connections and layers are not match\n");
		printf("cs: %lu, ls: %d\n",cs.sorted_cs.size(), ls.count());
		exit(0);
	}
	for(auto i:cs.sorted_cs){
		if (!ls.exist(i)){
			printf("error: not found layer %s\n",i.c_str());
			exit(0);
		}
	}
}
void Net004::show(){
	printf("[net] %s\n",name.c_str());
	int total_pm = 0;
	for(const auto i : cs.sorted_cs){
		ls.show(i);
		int pm = ls.parameter_number(i);
		total_pm += pm;
		printf("parameter number: %d\n",pm);
	}
	printf("total parameter number: %d\n",total_pm);
}
void Net004::setup(){
	map<string, int> ins,outs;
	cs.indegrees(ins);
	cs.outdegrees(outs);

	for(const auto& i : cs.sorted_cs){
		Layer* l0 = ls[i];
		//printf("+");
		//l0->show();
		if(outs.find(i) == outs.end()) continue;
		for (const auto& j : cs[i]){
			Layer* l1 = ls[j];
			//l1->show();
			l0->connect2(*l1);
			if(ins[j] == 0){
				printf("error: should not reach this line\n");
				exit(0);
			}
			ins[j] -= 1;
			if(ins[j] == 0){
				l1->setup();
			}
		}
	}
}
