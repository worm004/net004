#include "stdlib.h"
#include "BaseRun.h"
Run::Run(){
}
Run::Run(const JsonValue& j){
	j_ = j;
	const auto& o = j.jobj.at("attrs").jobj;
	type = o.at("type").jv.s;
	name = o.at("name").jv.s;
	if(o.find("iter") != o.end())
		iter = o.at("iter").jv.d;
	if(o.find("iter_interval") != o.end())
		iter_interval = o.at("iter_interval").jv.d;
}
void Run::show()const{
	printf("(type name) %s %s\n",type.c_str(),name.c_str());
	printf("  (iter iter_interval) %d %d\n",iter,iter_interval);
}
void Run::check(const Net004& net)const{
	if((name.size() == 0) || (type.size() == 0)){
		printf("name and type cannot be empty\n");
		exit(0);
	}
}
