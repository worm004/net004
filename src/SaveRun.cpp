#include "stdlib.h"
#include <sys/types.h>
#include <sys/stat.h>
#include "SaveRun.h"
using namespace std;
SaveRun::SaveRun(){ }
SaveRun::SaveRun(const JsonValue& j):Run(j){
	const auto& o = j.jobj.at("attrs").jobj;
	dir = o.at("dir").jv.s;
}
void SaveRun::show()const{
	Run::show();
	printf("  (dir) %s\n",dir.c_str());
}
void SaveRun::check(const Net004& net)const{
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
}
