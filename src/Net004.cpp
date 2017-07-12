#include "stdlib.h"
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
	for(const auto i : cs.sorted_cs)
		ls.show(i);
}
