#include "stdlib.h"
#include <sys/types.h>
#include <sys/stat.h>
#include "SaveRun.h"
#include "Parser.h"
using namespace std;
SaveRun::SaveRun(){ }
SaveRun::SaveRun(const JsonValue& j):Run(j){
	if(omit) return;
	const auto& o = j.jobj.at("attrs").jobj;
	prefix = o.at("prefix").jv.s;
	dir = o.at("dir").jv.s;
}
void SaveRun::show()const{
	if(omit) return;
	Run::show();
	printf("  (prefix) %s\n",prefix.c_str());
	printf("  (dir) %s\n",dir.c_str());
}
void SaveRun::check(const Net004& net)const{
	if(omit) return;
	Run::check(net);
	struct stat info;
	if( stat( dir.c_str(), &info ) != 0 ){
		printf( "cannot access save folder %s\n", dir.c_str() );
		exit(0);
	}
	else if( !(info.st_mode & S_IFDIR) ){
		printf( "save folder %s is not a directory\n", dir.c_str() );
		exit(0);
	}
}
void SaveRun::operator()(Net004& net, int cur){
	if(omit) return;
	if(cur%iter_interval != 0) return;
	string path = dir + "/" + prefix + ::to_string(cur)+".data";
	printf("[iter %d] [save] [model %s]\n",cur,path.c_str());
	ModelParser p;
	p.write_model(path,&net);
}
