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
	int total_pm = 0, total_im = 0, total_om = 0;
	for(const auto i : cs.sorted_cs){
		ls.show(i);
		int pm = ls.parameter_number(i);
		int im = ls.input_parameter_number(i);
		int om = ls.output_parameter_number(i);
		total_pm += pm;
		total_im += im;
		total_om += om;
		if(pm != 0) printf("\tparameter number: %d\n",pm);
		if(im != 0) printf("\tinput parameter number: %d\n",im);
		if(om != 0) printf("\toutput parameter number: %d\n",om);
	}
	printf("total parameter number: %d (%.2f mb)\n",total_pm, sizeof(float)*total_pm/1024.0f/1024.0f);
	printf("total input parameter number (include dif): %d (%.2f mb)\n", total_im * 2, sizeof(float) * 2 * total_im/1024.0f/1024.0f);
	printf("total output parameter number (include dif): %d (%.2f mb)\n",total_om * 2, sizeof(float) * 2 * total_om/1024.0f/1024.0f);
}
void Net004::setup(){
	map<string, int> ins,outs;
	cs.indegrees(ins);
	cs.outdegrees(outs);

	for(const auto& i : cs.sorted_cs){
		if(outs.find(i) == outs.end()) continue;
		Layer* l0 = ls[i];
		for (const auto& j : cs[i]){
			Layer* l1 = ls[j];
			l0->connect2(*l1);
			ins[j] -= 1;
			if(ins[j] < 0){
				printf("error: should not reach this line\n");
				exit(0);
			} else if(ins[j] == 0) l1->setup();
		}
	}
}
void Net004::forward(){
	for(const auto& i : cs.sorted_cs) 
		ls[i]->forward();
}
void Net004::backward(){
	for(int i=cs.sorted_cs.size()-1; i>=0;--i)
		ls[cs.sorted_cs[i]]->backward();
}
